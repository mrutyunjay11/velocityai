import time
from typing import Optional, Dict
from velocityai.agents.roles import (
    ArchitectAgent,
    BrowserAgent,
    DeveloperAgent,
    ReviewerAgent,
    SynthesizerAgent,
    OutlinerAgent,
    StorytellerAgent,
    EditorAgent
)
from velocityai.tools.browser import BrowserTool
from velocityai.tokenizer import Tokenizer
from velocityai.nn.transformer import LlamaModel

class AgentTeam:
    """
    Multi-Agent Collaborative Orchestrator.
    Coordinates specialized agents across 3 intelligent pipelines:
    - Creative/Narrative flow: Outliner -> Storyteller -> Editor
    - Research/Docs flow: BrowserResearcher -> Synthesizer -> Reviewer (Fact-Checker)
    - Coding/Implementation flow: BrowserResearcher -> Architect -> Developer -> Reviewer
    """
    def __init__(
        self,
        allow_internet: bool = False,
        plan_tokens: int = 350,
        code_tokens: int = 750,
        review_tokens: int = 600
    ):
        self.browser_tool = BrowserTool(allow_internet=allow_internet)
        self.architect = ArchitectAgent(max_tokens=plan_tokens)
        self.browser = BrowserAgent(browser_tool=self.browser_tool, max_tokens=400)
        self.synthesizer = SynthesizerAgent(max_tokens=plan_tokens)
        self.developer = DeveloperAgent(max_tokens=code_tokens)
        self.reviewer = ReviewerAgent(max_tokens=review_tokens)
        self.outliner = OutlinerAgent(max_tokens=plan_tokens)
        self.storyteller = StorytellerAgent(max_tokens=code_tokens)
        self.editor = EditorAgent(max_tokens=review_tokens)

    def set_internet_permission(self, allowed: bool):
        """Dynamically toggles internet permission for all agents."""
        self.browser_tool.allow_internet = allowed
        self.browser.browser_tool.allow_internet = allowed

    def _classify_intent(self, query: str) -> str:
        """
        Classifies user request into one of three specialized collaborative pipelines:
        1. 'creative': Creative storytelling, narrative prose, poetry, worldbuilding.
        2. 'research': Fact-finding, documentation search, explanations, technical overviews.
        3. 'coding': Software architecture, code implementation, debugging, refactoring.
        """
        q = query.lower()

        # Explicit code/engineering intent
        explicit_coding_keywords = (
            "write code", "implement ", "build an app", "create a function", "create a script",
            "create an app", "create a class", "develop an app", "make a program", "code for ",
            "write a python", "write a function", "write a class", "write a c++",
            "refactor ", "debug ", "optimize the code", "fix the bug", "pull request"
        )
        if any(kw in q for kw in explicit_coding_keywords):
            return "coding"

        # Creative narrative writing intent
        creative_keywords = (
            "story", "stories", "tale", "tales", "poem", "poems", "poetry",
            "narrative", "novel", "fiction", "sci-fi", "fantasy", "screenplay",
            "fable", "write a book", "short story", "creative writing", "mythology"
        )
        if any(kw in q for kw in creative_keywords):
            return "creative"

        # Research and technical documentation intent
        research_keywords = (
            "search", "find", "what is", "what are", "how does", "explain", "overview",
            "updates", "syntax", "difference", "docs", "documentation", "features",
            "tell me about", "list ", "summary", "summarize", "history of", "compare"
        )
        if any(kw in q for kw in research_keywords):
            return "research"

        # Default to coding if general programming or technical words appear, else research
        if any(kw in q for kw in ("code", "python", "script", "function", "algorithm", "class", "api", "database")):
            return "coding"

        return "research"

    def run(
        self,
        model: LlamaModel,
        tokenizer: Tokenizer,
        user_request: str,
        device: str = "cpu",
        sparse: bool = False,
        sparse_threshold: float = -3.5,
        stream: bool = True
    ) -> Dict[str, str]:
        """
        Executes the collaborative multi-agent pipeline with intelligent intent routing.
        """
        t0 = time.time()
        intent = self._classify_intent(user_request)

        if intent == "creative":
            flow_type = "📖 CREATIVE & NARRATIVE WRITING"
        elif intent == "research":
            flow_type = "🔬 RESEARCH & DOCUMENTATION SYNTHESIS"
        else:
            flow_type = "💻 SOFTWARE ENGINEERING & CODING"

        print("\n" + "=" * 65)
        print("  🤝 VelocityAI Multi-Agent Team Collaboration Activated")
        print(f"  Task Type: {flow_type}")
        net_status = "🌐 INTERNET ALLOWED" if self.browser_tool.allow_internet else "🔒 OFFLINE / PRIVATE"
        print(f"  Status:    {net_status}")
        print("=" * 65 + "\n")

        docs_summary = ""

        # =========================================================================
        # BRANCH A: RESEARCH & DOCUMENTATION SYNTHESIS PIPELINE
        # =========================================================================
        if intent == "research":
            # Step 1: Live Web Search / Docs Auditor
            needs_web = any(kw in user_request.lower() for kw in ("search", "latest", "update", "version", "docs", "api", "release", "feature"))
            if self.browser_tool.allow_internet or needs_web:
                docs_summary = self.browser.search_and_summarize(
                    model=model,
                    tokenizer=tokenizer,
                    query=user_request,
                    device=device
                )

            # Step 2: Technical Synthesizer
            print(self.synthesizer.banner())
            synth_context = f"Verified Source Material:\n{docs_summary}" if docs_summary and not docs_summary.startswith("[") else None
            synth_task = f"Synthesize a structured, accurate, and comprehensive answer for: {user_request}"
            synth_output = self.synthesizer.step(
                model=model,
                tokenizer=tokenizer,
                task=synth_task,
                context=synth_context,
                device=device,
                sparse=sparse,
                sparse_threshold=sparse_threshold,
                stream=stream
            )
            if stream:
                print()
            print(self.synthesizer.footer())

            # Step 3: Reviewer / Fact-Checker
            print(self.reviewer.banner())
            review_context = f"Synthesized Draft:\n{synth_output}"
            if docs_summary and not docs_summary.startswith("["):
                review_context += f"\n\nSource Documentation:\n{docs_summary}"
            review_task = (
                f"Verify and audit this response for: {user_request}. "
                f"Ensure complete factual consistency with the documentation, remove any speculative claims, and format cleanly."
            )
            final_output = self.reviewer.step(
                model=model,
                tokenizer=tokenizer,
                task=review_task,
                context=review_context,
                device=device,
                sparse=sparse,
                sparse_threshold=sparse_threshold,
                stream=stream
            )
            if stream:
                print()
            print(self.reviewer.footer())

            total_time = time.time() - t0
            print("-" * 65)
            print(f"✨ Research Synthesis Completed in {total_time:.2f}s!")
            print("-" * 65 + "\n")

            return {
                "docs": docs_summary,
                "synthesis": synth_output,
                "final": final_output,
                "duration": total_time
            }

        # =========================================================================
        # BRANCH B: CREATIVE & NARRATIVE WRITING PIPELINE
        # =========================================================================
        elif intent == "creative":
            # Step 1: Narrative Outliner (Plot, Character, Worldbuilding)
            print(self.outliner.banner())
            outline_task = f"Create a structured, inspiring narrative outline for: {user_request}"
            outline = self.outliner.step(
                model=model,
                tokenizer=tokenizer,
                task=outline_task,
                device=device,
                sparse=sparse,
                sparse_threshold=sparse_threshold,
                stream=stream
            )
            if stream:
                print()
            print(self.outliner.footer())

            # Step 2: Storyteller (Vivid Prose Narrative)
            print(self.storyteller.banner())
            story_context = f"Narrative Outline & Character Blueprint:\n{outline}"
            story_task = f"Write the complete, captivating, and vivid story based on the outline for: {user_request}"
            draft_story = self.storyteller.step(
                model=model,
                tokenizer=tokenizer,
                task=story_task,
                context=story_context,
                device=device,
                sparse=sparse,
                sparse_threshold=sparse_threshold,
                stream=stream
            )
            if stream:
                print()
            print(self.storyteller.footer())

            # Step 3: Literary Editor (Refinement, Tone, Final Polish)
            print(self.editor.banner())
            edit_context = f"Draft Story:\n{draft_story}\n\nOriginal Outline:\n{outline}"
            edit_task = (
                f"Polish and refine this story for: {user_request}. "
                f"Elevate sensory descriptions, improve pacing and emotional depth, eliminate awkward repetitions, and present the finalized story."
            )
            final_story = self.editor.step(
                model=model,
                tokenizer=tokenizer,
                task=edit_task,
                context=edit_context,
                device=device,
                sparse=sparse,
                sparse_threshold=sparse_threshold,
                stream=stream
            )
            if stream:
                print()
            print(self.editor.footer())

            total_time = time.time() - t0
            print("-" * 65)
            print(f"✨ Creative Narrative Completed in {total_time:.2f}s!")
            print("-" * 65 + "\n")

            return {
                "outline": outline,
                "story": draft_story,
                "final": final_story,
                "duration": total_time
            }

        # =========================================================================
        # BRANCH C: SOFTWARE ENGINEERING & CODING PIPELINE
        # =========================================================================
        # Step 1: Optional Web Search for APIs/Packages
        needs_web = any(kw in user_request.lower() for kw in ("library", "api", "version", "docs", "latest", "package", "gui", "web"))
        if self.browser_tool.allow_internet and needs_web:
            docs_summary = self.browser.search_and_summarize(
                model=model,
                tokenizer=tokenizer,
                query=user_request,
                device=device
            )

        # Step 2: Architect Agent (System Blueprint)
        print(self.architect.banner())
        plan = self.architect.step(
            model=model,
            tokenizer=tokenizer,
            task=f"Create a concise technical implementation plan for: {user_request}",
            device=device,
            sparse=sparse,
            sparse_threshold=sparse_threshold,
            stream=stream
        )
        if stream:
            print()
        print(self.architect.footer())

        # Step 3: Developer Agent (Implementation)
        print(self.developer.banner())
        dev_context = f"Architectural Blueprint:\n{plan}"
        if docs_summary and not docs_summary.startswith("["):
            dev_context += f"\n\nVerified Technical Documentation:\n{docs_summary}"

        dev_task = f"Write the clean, complete, production-grade code implementation for: {user_request}"
        code_output = self.developer.step(
            model=model,
            tokenizer=tokenizer,
            task=dev_task,
            context=dev_context,
            device=device,
            sparse=sparse,
            sparse_threshold=sparse_threshold,
            stream=stream
        )
        if stream:
            print()
        print(self.developer.footer())

        # Step 4: Reviewer Agent (Code Audit & Verification)
        print(self.reviewer.banner())
        review_context = f"Initial Code Implementation:\n{code_output}"
        review_task = (
            f"Review this code carefully for: {user_request}. "
            f"Fix any syntax errors, non-existent methods, and missing imports. "
            f"Do NOT convert simple functions into unnecessary OOP classes or overcomplicate the design. "
            f"Output the finalized, working production code in ```python."
        )
        final_output = self.reviewer.step(
            model=model,
            tokenizer=tokenizer,
            task=review_task,
            context=review_context,
            device=device,
            sparse=sparse,
            sparse_threshold=sparse_threshold,
            stream=stream
        )
        if stream:
            print()
        print(self.reviewer.footer())

        total_time = time.time() - t0
        print("-" * 65)
        print(f"✨ Multi-Agent Team Completed in {total_time:.2f}s!")
        print("-" * 65 + "\n")

        return {
            "plan": plan,
            "docs": docs_summary,
            "code": code_output,
            "final": final_output,
            "duration": total_time
        }

