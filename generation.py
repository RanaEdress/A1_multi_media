
import re
import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

MAX_CONTEXT_CHARS = 4000   
MAX_NEW_TOKENS    = 450    


class AnswerGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        print(f"Loading VLM: {model_name}")
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=bnb,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        ).eval()
        print(f" Qwen2-VL-2B loaded (4-bit, max_new_tokens={MAX_NEW_TOKENS})")

    def _build_page_ref(self, source_pages: list) -> str:
        if not source_pages:
            return "unknown source"
        parts = []
        for item in source_pages[:4]:
            if isinstance(item, tuple):
                page_num, source_doc = item
            else:
                page_num, source_doc = item, "doc"
            parts.append(f"p.{page_num} ({source_doc})")
        return ", ".join(parts)

    @staticmethod
    def _detect_target_doc(query: str):
        if re.search(r"\b(imf|weo|world economic outlook)\b", query, re.IGNORECASE):
            return "imf_weo"
        if re.search(r"\b(world bank)\b", query, re.IGNORECASE):
            return "world_bank"
        return None

    @staticmethod
    def _smart_trim_context(context: str, max_chars: int) -> str:
       
        if len(context) <= max_chars:
            return context

        sections = context.split("\n\n---\n\n")
        kept = []
        total = 0
        for sec in sections:
            sec_len = len(sec)
            if total + sec_len + 7 <= max_chars:  
                kept.append(sec)
                total += sec_len + 7
            else:
                remaining = max_chars - total - 7
                if remaining > 200:
                    kept.append(sec[:remaining] + "\n[...trimmed...]")
                break
        return "\n\n---\n\n".join(kept)

    @staticmethod
    def _extract_table_highlights(context: str, query: str, target_doc=None) -> str:
        query_words = set(re.findall(r"\w+", query.lower()))
        stop = {"what", "the", "in", "for", "are", "is", "a", "an", "of",
                "and", "to", "do", "does", "shown", "show", "figure", "table",
                "imf", "weo", "world", "bank", "report", "appear", "appears"}
        query_words -= stop
        if not query_words:
            return ""

        highlights = []
        for section in re.split(r"\n\n---\n\n", context):
            if not re.search(r"\|\s*TABLE\s*\]", section, re.IGNORECASE):
                continue
            if target_doc and target_doc not in section.lower():
                continue
            lines = [l.strip() for l in section.splitlines() if l.strip()]
            if len(lines) < 2:
                continue
            header_line = lines[0]
            data_lines  = lines[1:]
            col_header  = data_lines[0] if data_lines else ""
            matched_rows = [
                row for row in data_lines[1:]
                if any(w in row.lower() for w in query_words)
            ]
            if matched_rows:
                highlights.append(
                    f"{header_line}\n{col_header}\n" + "\n".join(matched_rows[:8])
                )

        return ("RELEVANT TABLE DATA:\n" + "\n\n".join(highlights) + "\n\n") if highlights else ""

    @staticmethod
    def _extract_caption_highlights(context: str, query: str, target_doc=None) -> str:
        highlights = []
        for section in re.split(r"\n\n---\n\n", context):
            if not re.search(r"\|\s*IMAGE_CAPTION\s*\]", section, re.IGNORECASE):
                continue
            if target_doc and target_doc not in section.lower():
                continue
            lines = [l.strip() for l in section.splitlines() if l.strip()]
            if lines:
                caption_text = " ".join(lines[1:]) if len(lines) > 1 else lines[0]
                highlights.append(caption_text)
        return ("FIGURE/CHART CAPTIONS FOUND:\n" + "\n".join(highlights) + "\n\n") if highlights else ""

    def _build_prompt(self, query: str, context: str, page_ref: str) -> str:
        ctx = self._smart_trim_context(context, MAX_CONTEXT_CHARS)
        ctx = re.sub(r"\n{3,}", "\n\n", ctx).strip()

        target_doc = self._detect_target_doc(query)

        caption_highlight = ""
        if re.search(r"\b(caption|figure|chart|box|fig\.?)\b", query, re.IGNORECASE):
            caption_highlight = self._extract_caption_highlights(context, query, target_doc)

        table_highlight = ""
        if re.search(
            r"\b(rate|percent|%|growth|inflation|gdp|projection|output|"
            r"figure|table|number|amount)\b",
            query, re.IGNORECASE
        ):
            table_highlight = self._extract_table_highlights(context, query, target_doc)

        injected = caption_highlight + table_highlight

        prompt = (
    "You are a precise document QA assistant.\n"
    "Answer the QUESTION using ONLY the DOCUMENT EXCERPTS below.\n\n"

    "Rules:\n"
    "1. Use ONLY the provided excerpts.\n"
    "2. If a table is present, extract EXACT values from it.\n"
    "3. Do NOT summarize tables — copy the relevant numbers.\n"
    "4. If numbers are missing, say 'Not found'.\n"
    "5. Answer in 1-2 precise sentences.\n"
    f"6. End with exactly: [Source: {page_ref}]\n"
    f"7. If the answer is not found write: "
    f"'Not found in provided context. [Source: {page_ref}]'\n\n"

    f"{injected}"
    f"DOCUMENT EXCERPTS:\n{ctx}\n\n"
    f"QUESTION: {query}\n\n"
    "ANSWER:"
)
        return prompt

    @staticmethod
    def _strip_prompt_echo(answer: str) -> str:
        if "ANSWER:" in answer:
            answer = answer.split("ANSWER:")[-1]
        if "Rules:" in answer:
            answer = answer.split("Rules:")[-1]
        return answer.strip()

    @staticmethod
    def _clean_answer(answer: str) -> str:
        answer = re.sub(r"```.*?```", "", answer, flags=re.DOTALL)
        answer = re.sub(r"`", "", answer)
        answer = re.sub(r"\n{2,}", " ", answer)
        answer = re.sub(r"  +", " ", answer)
        return answer.strip()

    @torch.no_grad()
    def generate(
        self,
        query: str,
        context: str,
        visual_pages: list = None,
        source_pages: list = None,
    ) -> str:
        torch.cuda.empty_cache()

        source_pages = source_pages or []
        page_ref     = self._build_page_ref(source_pages)
        prompt_text  = self._build_prompt(query, context, page_ref)
        content = []
        if visual_pages:
            for vp in visual_pages[:3]:
                _, _, img = vp
                resized = img.copy()
                resized.thumbnail((384, 384))   
                content.append({"type": "image", "image": resized})
        content.append({"type": "text", "text": prompt_text})

        messages   = [{"role": "user", "content": content}]
        text_input = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            repetition_penalty=1.1,
        )
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, generated_ids)
        ]
        answer = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        answer = self._strip_prompt_echo(answer)
        answer = self._clean_answer(answer)
       
        clean_check = re.sub(r"\[Source[^\]]*\]", "", answer, flags=re.IGNORECASE).strip()
        if not clean_check or len(clean_check.split()) < 4:
            return (
                f"The retrieved context does not contain a clear answer to this question. "
                f"[Source: {page_ref}]"
            )

        citation_pattern = re.compile(r"\[Source[s]?\s*:", re.IGNORECASE)
        if not citation_pattern.search(answer):
            answer = re.sub(r"\[Source[^\]]*\]?", "", answer, flags=re.IGNORECASE).strip()
            answer = f"{answer} [Source: {page_ref}]"

        return answer
