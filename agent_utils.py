# agent_utils.py

def load_agent_prompt(filepath: str = "agent_prompt.txt") -> str:
    with open(filepath, "r") as f:
        return f.read()

def format_agent_prompt(jd: str, resume: str, templates: list[str]) -> str:
    prompt_template = load_agent_prompt()
    return prompt_template.format(
        jd=jd,
        resume=resume,
        template_1=templates[0] if len(templates) > 0 else "",
        template_2=templates[1] if len(templates) > 1 else "",
        template_3=templates[2] if len(templates) > 2 else "",
    )
