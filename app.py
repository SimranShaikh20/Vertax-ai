"""Main entry point for the app.

This app is generated based on your prompt in Vertex AI Studio using
Google GenAI Python SDK (https://googleapis.github.io/python-genai/) and
Gradio (https://www.gradio.app/).

You can customize the app by editing the code in Cloud Run source code editor.
You can also update the prompt in Vertex AI Studio and redeploy it.
"""

import base64
from google import genai
from google.genai import types
import gradio as gr
import utils


def generate(
    message,
    history: list[gr.ChatMessage],
    request: gr.Request
):
  """Function to call the model based on the request."""

  validate_key_result = utils.validate_key(request)
  if validate_key_result is not None:
    yield validate_key_result
    return

  client = genai.Client(
      vertexai=True,
      project="qwiklabs-gcp-02-ee506ee791ec",
      location="global",
  )
  msg1_text1 = types.Part.from_text(text=f"""You are an expert AI assistant for an insurance underwriting department.
Your primary goal is to help underwriters by accurately and concisely summarizing client information and highlighting potential risk factors.
Maintain a professional and objective tone.
Focus only on the information provided in the prompt. Do not invent details.""")
  msg2_text1 = types.Part.from_text(text=f"""## Defining the Parameters

Okay, I see the task. My primary responsibility is to function as an AI assistant for insurance underwriting. That means I'll be handling core tasks, likely involving risk assessment and policy generation. It's important to understand the full scope of the role.


### Analyzing the Brief

I understand the requirements. I will summarize client data while focusing on risks, maintaining a professional, objective tone. I will avoid generating any new details. I will await the input.""")
  msg2_text2 = types.Part.from_text(text=f"""Understood. I am ready to assist. Please provide the client information you wish for me to summarize and analyze for potential risk factors. I will adhere strictly to the information provided and maintain an objective, professional tone.""")
  msg3_text1 = types.Part.from_text(text=f"""Customer Notes for 'SafeHarbor Warehousing':
\"The applicant is seeking coverage for their 50,000 sq ft warehouse. The business is 5 years old. The building is a concrete tilt-up structure, originally built in 2010. They store a variety of non-hazardous dry goods.
Fire safety measures include a full sprinkler system, a centrally monitored fire alarm, and documented annual inspections by a certified third party.
Security measures include a 24/7 centrally monitored burglar alarm, comprehensive security camera coverage of the interior and exterior, a fully fenced perimeter, and nightly patrols by a contracted security guard service.
The company reports no major property or liability losses in their 5-year history. They have specifically asked to ensure their new automated shelving and retrieval system, installed last month, is adequately covered under the policy.\"

Your Task:
1. Briefly summarize the key details of the 'SafeHarbor Warehousing' business and its existing safety measures.
2. Based *only* on the notes provided, identify any immediate questions an underwriter should ask or potential risk factors they should consider further.
Present the summary first, then the questions/risk factors as bullet points.""")


  model = "gemini-2.5-flash-preview-05-20"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
    types.Content(
      role="model",
      parts=[
        msg2_text1,
        msg2_text2
      ]
    ),
    types.Content(
      role="user",
      parts=[
        msg3_text1
      ]
    ),
  ]

  for prev_msg in history:
    role = "user" if prev_msg["role"] == "user" else "model"
    parts = utils.get_parts_from_message(prev_msg["content"])
    if parts:
      contents.append(types.Content(role=role, parts=parts))

  if message:
    contents.append(
        types.Content(role="user", parts=utils.get_parts_from_message(message))
    )

  generate_content_config = types.GenerateContentConfig(
      temperature=1,
      top_p=1,
      seed=0,
      max_output_tokens=65535,
      safety_settings=[
          types.SafetySetting(
              category="HARM_CATEGORY_HATE_SPEECH",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_DANGEROUS_CONTENT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
              threshold="OFF"
          ),
          types.SafetySetting(
              category="HARM_CATEGORY_HARASSMENT",
              threshold="OFF"
          )
      ],
  )

  results = []
  for chunk in client.models.generate_content_stream(
      model=model,
      contents=contents,
      config=generate_content_config,
  ):
    if chunk.candidates and chunk.candidates[0] and chunk.candidates[0].content:
      results.extend(
          utils.convert_content_to_gr_type(chunk.candidates[0].content)
      )
      if results:
        yield results

with gr.Blocks(theme=utils.custom_theme) as demo:
  with gr.Row():
    gr.HTML(utils.public_access_warning)
  with gr.Row():
    with gr.Column(scale=1):
      with gr.Row():
        gr.HTML("<h2>Welcome to Vertex AI GenAI App!</h2>")
      with gr.Row():
        gr.HTML("""This prototype was built using your Vertex AI Studio prompt.
            Follow the steps and recommendations below to begin.""")
      with gr.Row():
        gr.HTML(utils.next_steps_html)

    with gr.Column(scale=2, variant="panel"):
      gr.ChatInterface(
          fn=generate,
          title="Insurance Underwriting Assistant",
          type="messages",
          multimodal=True,
      )
  demo.launch(show_error=True)