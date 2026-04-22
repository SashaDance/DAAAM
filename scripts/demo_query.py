"""Interactive Scene Graph Q&A demo.

Loads a DSG, sets up the SceneUnderstandingAgent, and runs a REPL
for free-form questions against the scene graph.
"""
import os
import click
import spark_dsg as sdsg
from pathlib import Path

from daaam.utils.evaluation import preprocess_scene_graph, START_TIMES

from daaam.scene_understanding.services import SceneUnderstandingAgent
from daaam.scene_understanding.config import SceneUnderstandingConfig
from daaam.scene_understanding.models import TextResponse
from daaam.utils.logging import ConsoleLogger


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4.1-mini"


def configure_openrouter_env() -> None:
	"""Default the demo to OpenRouter while still using OPENAI_API_KEY."""
	os.environ.setdefault("OPENAI_BASE_URL", OPENROUTER_BASE_URL)
	os.environ.setdefault("OPENROUTER_SITE_URL", "https://github.com/MIT-SPARK/DAAAM")
	os.environ.setdefault("OPENROUTER_APP_NAME", "DAAAM demo_query")


@click.command()
@click.option("--dsg-path", type=click.Path(exists=True), required=True)
@click.option("--seq-id", type=int, required=True, help="CODA sequence ID (for START_TIMES lookup)")
@click.option("--model-name", type=str, default=OPENROUTER_DEFAULT_MODEL)
@click.option(
	"--sentence-embedding-model-name",
	type=str,
	default="sentence-transformers/sentence-t5-large",
	help="Sentence embedding model used for query-time semantic search.",
)
@click.option(
	"--clip-model-name",
	type=str,
	default="ViT-L-14",
	help="CLIP model name used for query-time semantic search.",
)
@click.option(
	"--clip-backend",
	type=click.Choice(["openclip", "pe"]),
	default="openclip",
	help="CLIP backend used for query-time semantic search.",
)
def main(
	dsg_path: str,
	seq_id: int,
	model_name: str,
	sentence_embedding_model_name: str,
	clip_model_name: str,
	clip_backend: str,
):
	configure_openrouter_env()

	if not os.environ.get("OPENAI_API_KEY"):
		raise click.ClickException(
			"OPENAI_API_KEY is not set. Export your OpenRouter key first, for example: "
			"export OPENAI_API_KEY=<YOUR_OPENROUTER_KEY>"
		)

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
	config.tool_config.sentence_embedding_model_name = sentence_embedding_model_name
	config.tool_config.clip_model_name = clip_model_name
	config.tool_config.clip_backend = clip_backend
	agent = SceneUnderstandingAgent(config, ConsoleLogger())
	agent.update_scene_graph(sg)

	n_nodes = sum(1 for _ in sg.nodes)
	print(f"Scene graph loaded ({n_nodes} nodes). Model: {model_name}")
	print(f"LLM endpoint: {os.environ['OPENAI_BASE_URL']}")
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
