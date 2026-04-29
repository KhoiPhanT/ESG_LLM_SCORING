"""
LLM Client — Giao tiếp với Qwen3:30b qua Ollama REST API.
Đây là module trung tâm của toàn bộ pipeline ESG.
Enhanced: expanded context window, stricter prompts for accuracy.
"""
import json
import requests
import re
import time


class OllamaClient:
    def __init__(self, model="qwen3:30b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self.chat_url = f"{base_url}/api/chat"

    def _call(self, messages, temperature=0.3, max_tokens=3072, retries=0):
        """Gửi request tới Ollama và trả về response text."""
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_ctx": 16384,
            },
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(self.chat_url, json=payload, timeout=180)
                resp.raise_for_status()
                content = resp.json()["message"]["content"]
                # Qwen3: strip <think>...</think> tags
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
                return content
            except Exception as e:
                if attempt < retries:
                    print(f"  [LLM retry {attempt+1}] {e}")
                    time.sleep(2)
                else:
                    print(f"  [LLM ERROR] {e}")
                    return None

    def _parse_json(self, text):
        """Trích xuất JSON từ response LLM (có thể lẫn text thừa hoặc bị cắt dở)."""
        if not text:
            return None
        # Tìm JSON block (lấy từ dấu { đầu tiên đến dấu } cuối cùng)
        match = re.search(r"(\{.*\})", text, re.DOTALL)
        if match:
            try:
                clean_json = match.group(1).strip()
                return json.loads(clean_json)
            except json.JSONDecodeError:
                pass

        # Fallback 1: JSON bị cắt dở do max_tokens — thử repair
        match2 = re.search(r"(\{.*)", text, re.DOTALL)
        if match2:
            truncated = match2.group(1).strip()
            # Try closing open strings and braces
            for fix in ['"}', '"}', '}', '"}']:
                try:
                    return json.loads(truncated + fix)
                except json.JSONDecodeError:
                    continue
            # Try extracting just answer from truncated JSON
            answer_match = re.search(r'"answer"\s*:\s*"([A-Z,]+)"', truncated)
            reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', truncated)
            evidence_match = re.search(r'"evidence"\s*:\s*"([^"]{10,})"?', truncated)
            if answer_match:
                return {
                    "answer": answer_match.group(1),
                    "selected_options": [x.strip() for x in answer_match.group(1).split(",")],
                    "reason": reason_match.group(1) if reason_match else "",
                    "evidence": evidence_match.group(1) if evidence_match else None,
                }

        # Fallback 2: thử parse toàn bộ
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    # ─── VNSI Question Answering ─────────────────────────────

    def ask_vnsi_question(self, context, question, options, q_id="", is_multi_select=False):
        """
        Trả lời 1 câu hỏi VNSI dựa trên context báo cáo.
        Returns: {"answer": "C", "selected_options": ["A"], "reason": "...", "evidence": "..."}
        """
        if is_multi_select:
            answer_format = '"selected_options": ["<one or more option letters>"], "answer": "<letters joined by comma>"'
            multi_instruction = """- This question allows MULTIPLE selections. Evaluate EACH criterion (A, B, C, D...) independently.
- Select ALL options that the company meets based on the evidence.
- Example: if company meets both C and D, return selected_options: ["C", "D"] and answer: "C,D"."""
        else:
            answer_format = '"answer": "<single letter A/B/C/D>", "selected_options": ["<same single letter>"]'
            multi_instruction = "- Select exactly ONE option that best matches the company's situation."

        prompt = f"""
You are an ESG analyst evaluating a Vietnamese company's annual report.
Answer the following VNSI assessment question based on the provided report context.
Please think step-by-step to answer accurately, but keep your reasoning concise to save tokens.

CRITICAL RULES:
- You MUST always select the most appropriate option letter(s). Do NOT answer "NULL".
- If you cannot find explicit evidence in the report, reason about what the ABSENCE of information implies for this specific question.
  For example: no mention of environmental policy likely means the company does not have one → select the corresponding negative option.
- Only use "NULL" as an absolute last resort if the question is completely inapplicable (e.g., manufacturing question for a bank).
{multi_instruction}

EVIDENCE QUALITY RULES:
- If the company has a clear official statement, policy, system, or certification (e.g., ISO) related to the question, treat it as VALID evidence and award points (+1), even if it lacks specific numbers.
- For VNSI scoring, companies are often credited if they officially mention tracking, managing, or having policies for a specific issue.
- Only penalize or select negative options (-1 or 0) if the report is COMPLETELY SILENT on the topic, or explicitly says they do not do it.
- The evidence quote must be a passage from the report that addresses the topic.

QUESTION [{q_id}]: {question}

OPTIONS:
{options}

REPORT CONTEXT:
{context[:30000]}

Respond with a single JSON object ONLY:
{{{answer_format}, "reason": "<short explanation in Vietnamese>", "evidence": "<short exact quote from report if found, else null>"}}"""

        messages = [{"role": "user", "content": prompt}]
        raw = self._call(messages, temperature=0.1, max_tokens=4096)
        result = self._parse_json(raw)
        if result and "answer" in result:
            if "selected_options" not in result or not result["selected_options"]:
                answer = str(result.get("answer", "")).strip()
                if answer.upper() == "NULL":
                    result["selected_options"] = []
                else:
                    result["selected_options"] = [x.strip() for x in answer.split(",") if x.strip()]
            return result
        # Fallback nếu parse thất bại hoặc LLM không trả về kết quả
        return {
            "answer": "NULL",
            "selected_options": [],
            "reason": "Không tìm thấy bằng chứng xác thực trong tài liệu (Fallback)", 
            "evidence": None
        }

    # ─── Screening Questions ─────────────────────────────────

    def ask_screening_question(self, context, question, q_id=""):
        """
        Trả lời câu hỏi sàng lọc (SL1-SL5): Có/Không.
        """
        prompt = f"""
You are auditing a Vietnamese company's annual report for ESG violations.
Answer this screening question with ONLY "A" (Yes, violation found) or "B" (No violation found).
Be conservative: only answer "A" if there is CLEAR evidence of violation.
Think step-by-step concisely to prevent reaching token limits.

QUESTION [{q_id}]: {question}

REPORT CONTEXT:
{context[:30000]}

Respond with JSON only:
{{"answer": "A or B", "reason": "<1 sentence in Vietnamese>"}}"""

        messages = [{"role": "user", "content": prompt}]
        raw = self._call(messages, temperature=0.1, max_tokens=2048)
        result = self._parse_json(raw)
        if result and "answer" in result:
            return result
        return {"answer": "B", "reason": "Không phát hiện vi phạm rõ ràng"}

    # ─── Entity Extraction ───────────────────────────────────

    def extract_esg_entities(self, context):
        """Trích xuất các chỉ số định lượng ESG từ đoạn báo cáo."""
        prompt = f"""
Extract quantitative ESG metrics from this Vietnamese company report section.
Think step-by-step concisely before returning the JSON output.
Return JSON with these fields (use null if not found):

{{"co2_emission": "value with unit or null",
 "energy_consumption": "value with unit or null",
 "water_usage": "value with unit or null",
 "waste_treated": "value with unit or null",
 "female_leadership_ratio": "percentage or null",
 "training_hours_per_employee": "number or null",
 "social_investment": "value with unit or null",
 "employee_count": "number or null",
 "rd_expense": "value with unit or null"}}

TEXT:
{context[:8000]}

JSON:"""
        messages = [{"role": "user", "content": prompt}]
        raw = self._call(messages, temperature=0.2, max_tokens=2048)
        result = self._parse_json(raw)
        return result or {}


if __name__ == "__main__":
    client = OllamaClient()
    # Quick test
    result = client.ask_vnsi_question(
        context="ACB đã xây dựng chiến lược phát triển bền vững với cam kết bảo vệ môi trường.",
        question="Công ty có chính sách liên quan tới quản lý các tác động môi trường?",
        options="A. Không có\nB. Có nhưng không công khai\nC. Có và công khai",
        q_id="E.1.1.1",
    )
    print("Test result:", json.dumps(result, ensure_ascii=False, indent=2))
