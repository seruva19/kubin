from dynamicprompts.generators import RandomPromptGenerator


def generate_prompt_from_wildcard(prompt):
    generator = RandomPromptGenerator()
    (prompt,) = generator.generate(prompt, num_images=1)
    return prompt
