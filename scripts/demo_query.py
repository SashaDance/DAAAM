"""Interactive Scene Graph Q&A demo.

Loads a DSG, sets up the SceneUnderstandingAgent, and runs a REPL
for free-form questions against the scene graph.
"""
import click
import spark_dsg as sdsg
from pathlib import Path

from daaam.utils.evaluation import preprocess_scene_graph, START_TIMES

from daaam.scene_understanding.services import SceneUnderstandingAgent
from daaam.scene_understanding.config import SceneUnderstandingConfig
from daaam.scene_understanding.models import TextResponse
from daaam.utils.logging import ConsoleLogger


@click.command()
@click.option("--dsg-path", type=click.Path(exists=True), required=True)
@click.option("--seq-id", type=int, required=True, help="CODA sequence ID (for START_TIMES lookup)")
@click.option("--model-name", type=str, default="gpt-5-mini")
def main(dsg_path: str, seq_id: int, model_name: str):
	assert seq_id in START_TIMES, f"Unknown seq_id {seq_id}. Valid: {sorted(START_TIMES.keys())}"

	sg = sdsg.DynamicSceneGraph.load(dsg_path)
	bg_yaml = Path(dsg_path).parent / "background_objects.yaml"
	sg = preprocess_scene_graph(
		sg,
		START_TIMES[seq_id],
		bg_yaml if bg_yaml.exists() else None,
	)

	config = SceneUnderstandingConfig(
		model_name=model_name,
		available_tools=[
			"get_matching_subjects",
			"get_objects_in_radius",
			"get_region_information",
			"get_agent_trajectory_information",
		],
	)
	agent = SceneUnderstandingAgent(config, ConsoleLogger())
	agent.update_scene_graph(sg)

	n_nodes = sum(1 for _ in sg.nodes)
	print(f"Scene graph loaded ({n_nodes} nodes). Model: {model_name}")
	print("Type your question (empty line or Ctrl+C to quit).\n")

	while True:
		try:
			question = input("> ").strip()
		except (EOFError, KeyboardInterrupt):
			print()
			break
		if not question:
			break

		answer, _, _ = agent.answer_query(TextResponse, question)
		print(f"\nAnswer:    {answer.answer}")
		print(f"Reasoning: {answer.reasoning}\n")


if __name__ == "__main__":
	main()
