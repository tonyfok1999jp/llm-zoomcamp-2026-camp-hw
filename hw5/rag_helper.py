from opentelemetry import trace

tracer = trace.get_tracer(__name__)

INSTRUCTIONS = '''
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
'''

PROMPT_TEMPLATE = '''
QUESTION: {question}

CONTEXT:
{context}
'''.strip()


class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='gpt-5.4-mini'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []

        for doc in search_results:
            lines.append(doc['filename'])
            lines.append(doc['content'])
            lines.append('')

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {'role': 'developer', 'content': self.instructions},
            {'role': 'user', 'content': prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text


class RAGTraced(RAGBase):

    def search(self, query, num_results=5):
        with tracer.start_as_current_span('search') as span:
            span.set_attribute('rag.query', query)
            span.set_attribute('rag.num_results_requested', num_results)

            search_results = super().search(query, num_results=num_results)

            span.set_attribute('rag.num_results_returned', len(search_results))
            return search_results

    def build_prompt(self, query, search_results):
        with tracer.start_as_current_span('build_prompt') as span:
            prompt = super().build_prompt(query, search_results)

            span.set_attribute('rag.prompt', prompt)
            return prompt

    def llm(self, prompt):
        with tracer.start_as_current_span('llm') as span:
            span.set_attribute('rag.model', self.model)
            span.set_attribute('rag.prompt', prompt)

            response = super().llm(prompt)

            span.set_attribute('rag.response', response.output_text)

            usage = response.usage
            span.set_attribute('input_tokens', usage.input_tokens)
            span.set_attribute('output_tokens', usage.output_tokens)

            return response

    def rag(self, query):
        with tracer.start_as_current_span('rag') as span:
            span.set_attribute('rag.query', query)

            answer = super().rag(query)

            span.set_attribute('rag.answer', answer)
            return answer
