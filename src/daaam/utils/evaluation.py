import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import click
import numpy as np


START_TIMES = {
	0: 1673884185.589118,
	3: 1673985684.662767,
	4: 1674087704.955325,
	6: 1674764557.790565,
	16: 1675881269.033398,
	21: 1676052119.125703,
	22: 1676064138.910544
}

def evaluate_output(qa_instance: Dict, predicted: Any) -> Dict:
	"""Evaluate the predicted output against ground truth and compute metrics."""
	out_error = {}
	q_type = qa_instance['type']

	if q_type == 'position':
		# Ground truth can be in different formats
		if isinstance(qa_instance['answers'], list) and len(qa_instance['answers']) > 0:
			if isinstance(qa_instance['answers'][0], dict) and 'position' in qa_instance['answers'][0]:
				answer = np.array(qa_instance['answers'][0]['position'])
			else:
				# Assume it's a direct position list
				answer = np.array(qa_instance['answers'][0] if isinstance(qa_instance['answers'][0], list) else qa_instance['answers'])
		elif isinstance(qa_instance['answers'], dict) and 'position' in qa_instance['answers']:
			answer = np.array(qa_instance['answers']['position'])
		else:
			answer = np.array([0, 0, 0])  # Default if no answer

		# Response objects from SceneUnderstandingAgent have 'answer' field
		if hasattr(predicted, 'answer'):
			pred_pos = np.array(predicted.answer if predicted.answer is not None else [0, 0, 0])
		elif isinstance(predicted, dict) and 'answer' in predicted:
			pred_pos = np.array(predicted['answer'])
		elif isinstance(predicted, dict) and 'position' in predicted:
			pred_pos = np.array(predicted['position'])
		else:
			pred_pos = np.array([0, 0, 0])

		try:
			dist = np.linalg.norm(answer - pred_pos)
			out_error['position_error'] = float(dist)
		except:
			out_error['position_error'] = float('inf')

	elif q_type == 'binary':
		# Get ground truth answer
		answer_text = None
		if isinstance(qa_instance['answers'], dict) and 'text' in qa_instance['answers']:
			# Format: {'text': ['Yes', 'Yes']}
			text_answers = qa_instance['answers']['text']
			if isinstance(text_answers, list):
				for ans in text_answers:
					if isinstance(ans, str) and ans.lower().strip() in ['yes', 'no']:
						answer_text = ans.lower().strip()
						break
		elif isinstance(qa_instance['answers'], list):
			# Alternative format: direct list
			for ans in qa_instance['answers']:
				if isinstance(ans, str) and ans.lower().strip() in ['yes', 'no']:
					answer_text = ans.lower().strip()
					break

		# Response objects from SceneUnderstandingAgent have 'answer' field
		if hasattr(predicted, 'answer'):
			pred_binary = str(predicted.answer).lower().strip() if predicted.answer is not None else 'unknown'
		elif isinstance(predicted, dict) and 'answer' in predicted:
			pred_binary = str(predicted['answer']).lower().strip()
		elif isinstance(predicted, dict) and 'binary' in predicted:
			pred_binary = str(predicted['binary']).lower().strip()
		else:
			pred_binary = 'unknown'

		if answer_text and pred_binary in ['yes', 'no']:
			correct = 1 if pred_binary == answer_text else 0
			out_error['binary_iscorrect'] = correct
			# Debug info
			out_error['binary_ground_truth'] = answer_text
			out_error['binary_predicted'] = pred_binary
		else:
			out_error['binary_iscorrect'] = 0
			# Debug info for failures
			out_error['binary_ground_truth'] = answer_text if answer_text else 'not_found'
			out_error['binary_predicted'] = pred_binary
			out_error['binary_debug'] = f"Answer format: {qa_instance.get('answers')}"

	elif q_type == 'time' or q_type == 'duration':
		# Get ground truth (in MINUTES, relative for "time" questions)
		answer = 0
		if isinstance(qa_instance.get('answers'), dict):
			# Format: {"text": ["9.43 minutes ago"], "time": 9.43}
			answer = float(qa_instance['answers'].get(q_type, 0))  # This is in minutes (relative for time)
		elif isinstance(qa_instance.get('answers'), list) and len(qa_instance['answers']) > 0:
			if isinstance(qa_instance['answers'][0], dict):
				answer = qa_instance['answers'][0].get(q_type, 0)
			else:
				answer = float(qa_instance['answers'][0]) if qa_instance['answers'] else 0

		# Response objects from SceneUnderstandingAgent have 'answer' field (absolute SECONDS from start)
		pred_time_seconds = 0
		if hasattr(predicted, 'answer'):
			pred_time_seconds = float(predicted.answer) if predicted.answer is not None else 0
		elif isinstance(predicted, dict) and 'answer' in predicted:
			pred_time_seconds = float(predicted['answer'])
		elif isinstance(predicted, dict) and q_type in predicted:
			pred_time_seconds = float(predicted[q_type])

		if q_type == 'time':
			# For "time" questions, we need to convert absolute to relative
			# Extract current time from question: "The current time is X seconds from start"
			current_time_seconds = 0
			question_text = qa_instance.get('question', '')
			time_match = re.search(r'current time is ([\d.]+) seconds from start', question_text)
			if time_match:
				current_time_seconds = float(time_match.group(1))

			# Convert to relative time: "X minutes ago"
			# Model gives absolute seconds, we need (current_time - event_time) / 60
			pred_time_minutes_ago = (current_time_seconds - pred_time_seconds) / 60.0

			dist = abs(answer - pred_time_minutes_ago)
			out_error[f'{q_type}_error'] = float(dist)

			# Store debugging info
			out_error[f'{q_type}_ground_truth_minutes_ago'] = answer
			out_error[f'{q_type}_predicted_minutes_ago'] = pred_time_minutes_ago
			out_error['current_time_seconds'] = current_time_seconds
			out_error['predicted_absolute_seconds'] = pred_time_seconds
		else:
			# For "duration" questions, just convert seconds to minutes
			pred_time_minutes = pred_time_seconds / 60.0

			dist = abs(answer - pred_time_minutes)
			out_error[f'{q_type}_error'] = float(dist)

			# Store debugging info
			out_error[f'{q_type}_ground_truth_minutes'] = answer
			out_error[f'{q_type}_predicted_minutes'] = pred_time_minutes

	elif q_type == 'text':
		# For text questions, just store the ground truth answer
		out_error['answer'] = qa_instance['answers']

	else:
		# Unknown question type
		out_error['error'] = f"Unknown question type: {q_type}"

	return out_error