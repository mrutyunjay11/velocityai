from typing import Optional, List, Dict
from velocityai.agents.agent import Agent
from velocityai.tools.browser import BrowserTool
from velocityai.tokenizer import Tokenizer
from velocityai.nn.transformer import LlamaModel

class ArchitectAgent(Agent):
    """Responsible for breaking down complex requirements into technical blueprints."""
    def __init__(self, max_tokens: int = 512):
        super().__init__(
            name="Architect",
            role="📋 Blueprint & System Design",
            system_prompt=(
                "You are an expert Lead Software Architect. "
                "Your objective is to analyze the user's requirements and produce a concise, high-level technical specification: "
                "1. Algorithm approach, 2. Function signatures and inputs/outputs, 3. Edge cases to handle. "
                "STRICT RULE: Do NOT write Python code, function implementations, or code blocks (no def, no ```). "
                "Provide only a concise, bulleted architectural specification. Keep it under 150 words."
            ),
            temperature=0.2,
            top_p=0.95,
            repetition_penalty=1.1,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[34m" # Blue
        )


class BrowserAgent(Agent):
    """Responsible for searching and retrieving live web documentation (permission-gated)."""
    def __init__(self, browser_tool: Optional[BrowserTool] = None, max_tokens: int = 220):
        super().__init__(
            name="BrowserResearcher",
            role="🌐 Web Documentation Auditor",
            system_prompt=(
                "You are a Senior Technical Researcher. "
                "Your objective is to extract at most 5 to 7 key distinct API updates or features from the web documentation. "
                "STRICT RULE: Stop after 5 to 7 bullet points. Do not produce an endless list."
            ),
            temperature=0.15,
            top_p=0.9,
            repetition_penalty=1.20,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[35m" # Magenta
        )
        self.browser_tool = browser_tool or BrowserTool(allow_internet=False)

    def search_and_summarize(
        self,
        model: LlamaModel,
        tokenizer: Tokenizer,
        query: str,
        device: str = "cpu"
    ) -> str:
        """Searches the web if authorized, fetches top result, and summarizes facts."""
        if not self.browser_tool.allow_internet:
            # Check if user grants permission
            allowed = self.browser_tool.check_or_request_permission(f"search docs for: {query}")
            if not allowed:
                return "[Offline Mode: Internet access disabled. Proceeding with local model knowledge.]"

        print(f"\n{self.banner()}")
        print(f"│ Searching web: '{query}'...")
        results = self.browser_tool.search(query, max_results=3)

        if not results or not results[0].get("url"):
            print(f"│ {results[0].get('snippet', 'No results found.')}")
            print(self.footer())
            return "[No external documentation found.]"

        top_url = results[0]["url"]
        print(f"│ Reading documentation from: {top_url[:60]}...")
        page_text = self.browser_tool.fetch_page(top_url, max_chars=2000)

        # Summarize key technical facts using local LLM
        prompt = (
            f"From the following web documentation, extract the top 5 to 7 most important updates for '{query}':\n\n"
            f"{page_text}\n\n"
            f"List strictly 5 to 7 distinct bullet points. Stop after 7 items."
        )
        summary = self.step(
            model=model,
            tokenizer=tokenizer,
            task=prompt,
            device=device,
            stream=False,
            auto_continue=False
        )
        print(f"│ Verified API Facts:\n│ " + summary.strip().replace("\n", "\n│ "))
        print(self.footer())
        return summary


class SynthesizerAgent(Agent):
    """Responsible for synthesizing research, documentation, and factual answers clearly and concisely."""
    def __init__(self, max_tokens: int = 450):
        super().__init__(
            name="Synthesizer",
            role="📝 Technical Synthesizer & Writer",
            system_prompt=(
                "You are an expert Technical Writer and Synthesizer. "
                "Your objective is to answer the user's specific inquiry with a structured, accurate explanation "
                "covering at most 5 to 7 distinct key points based on the verified technical documentation. "
                "STRICT RULE: Do not repeat identical points or list the same module multiple times. Keep each point unique."
            ),
            temperature=0.15,
            top_p=0.9,
            repetition_penalty=1.20,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[36m" # Cyan
        )


class DeveloperAgent(Agent):
    """Responsible for writing clean, modular, bug-free implementation."""
    def __init__(self, max_tokens: int = 750):
        super().__init__(
            name="Developer",
            role="💻 Software Engineer (Implementation)",
            system_prompt=(
                "You are an expert Senior Software Engineer. "
                "Your objective is to implement clean, production-grade, self-contained code based on the user's request. "
                "STRICT RULES: "
                "1. Write ONLY ONE complete, working Python implementation. Do NOT provide multiple alternative implementations. "
                "2. Do NOT copy the blueprint text, comments, or explanations. "
                "3. Start immediately with the imports or function definition inside ```python."
            ),
            temperature=0.15,
            top_p=0.95,
            repetition_penalty=1.15,
            code_mode=True,
            max_tokens=max_tokens,
            color_code="\033[32m" # Green
        )


class ReviewerAgent(Agent):
    """Responsible for scrutinizing code or technical reports, catching bugs/hallucinations, and providing the finalized solution."""
    def __init__(self, max_tokens: int = 550):
        super().__init__(
            name="Reviewer",
            role="🔍 Quality & Security Auditor",
            system_prompt=(
                "You are a Principal Code Reviewer and Quality Auditor. "
                "Carefully inspect the output to ensure it directly and accurately answers the user's prompt. "
                "1. If reviewing code: ensure it works correctly and has proper imports. "
                "   STRICT RULE: Do NOT convert simple functions into OOP classes or overcomplicate the design. Preserve the exact function name and signature requested. Output the clean, audited code in ```python. "
                "2. If reviewing a technical report or research: eliminate any duplicate or repetitive points, and format cleanly. "
                "STRICT RULE: Do not repeat identical points with different numbers. Keep each point unique."
            ),
            temperature=0.15,
            top_p=0.9,
            repetition_penalty=1.20,
            code_mode=True,
            max_tokens=max_tokens,
            color_code="\033[33m" # Yellow
        )


class OutlinerAgent(Agent):
    """Responsible for narrative structure, worldbuilding, and character outline in creative writing tasks."""
    def __init__(self, max_tokens: int = 450):
        super().__init__(
            name="Outliner",
            role="🎨 Narrative Architect & Worldbuilding",
            system_prompt=(
                "You are a Master Narrative Architect and Story Outliner. "
                "Your objective is to craft a compelling structural outline for the requested story or creative work. "
                "Define: 1. Setting & Atmosphere, 2. Protagonist & Key Characters, 3. Inciting Incident & Central Conflict, 4. Three-Act Narrative Arc (Beginning, Climax, Resolution). "
                "Keep the outline structured, inspiring, and concise. Do NOT write the full prose story yet."
            ),
            temperature=0.35,
            top_p=0.95,
            repetition_penalty=1.15,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[35m" # Magenta
        )


class StorytellerAgent(Agent):
    """Responsible for crafting immersive, vivid narrative prose based on the outline."""
    def __init__(self, max_tokens: int = 750):
        super().__init__(
            name="Storyteller",
            role="📖 Vivid Narrative & Scene Construction",
            system_prompt=(
                "You are an acclaimed Creative Writer and Novelist. "
                "Your objective is to write the complete, immersive narrative based on the structural outline. "
                "Focus on vivid sensory details, emotional depth, natural dialogue, and engaging pacing. "
                "Bring the world and characters to life. Write rich, captivating prose from start to finish."
            ),
            temperature=0.4,
            top_p=0.95,
            repetition_penalty=1.15,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[32m" # Green
        )


class EditorAgent(Agent):
    """Responsible for literary polish, prose rhythm, tone refinement, and delivery of the finalized story."""
    def __init__(self, max_tokens: int = 750):
        super().__init__(
            name="Editor",
            role="✍️ Literary Editor & Final Polish",
            system_prompt=(
                "You are a Senior Literary Editor and Prose Stylist. "
                "Carefully review and refine the story draft for maximum impact: "
                "1. Enhance prose rhythm, word choice, and evocative descriptions. "
                "2. Eliminate any repetitive phrases, awkward pacing, or abrupt transitions. "
                "3. Deliver the finalized, beautifully polished version of the story."
            ),
            temperature=0.25,
            top_p=0.95,
            repetition_penalty=1.15,
            code_mode=False,
            max_tokens=max_tokens,
            color_code="\033[36m" # Cyan
        )


