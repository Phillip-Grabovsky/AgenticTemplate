from state import State
import agents.example_agent as example_agent
import asyncio

async def main():
    state = State()
    state.question = "What is the capital of France?"
    myAgent = example_agent.ExampleAgent(state)
    await myAgent.answer_question()
    print(state.answer)

if __name__ == "__main__":
    asyncio.run(main())