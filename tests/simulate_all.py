import os
import time
import base64
import io
from typing import List, Dict, Any
from unittest.mock import MagicMock

# --- Mocks for Missing Dependencies ---
import sys
if "PIL" not in sys.modules:
    mock_pil = MagicMock()
    mock_img = MagicMock()
    mock_img.size = (1024, 1024)
    mock_img.mode = "RGB"
    mock_pil.Image.open.return_value = mock_img
    mock_pil.Image.new.return_value = mock_img
    sys.modules["PIL"] = mock_pil
    sys.modules["PIL.Image"] = mock_pil.Image
if "cv2" not in sys.modules:
    sys.modules["cv2"] = MagicMock()
if "pydub" not in sys.modules:
    sys.modules["pydub"] = MagicMock()
    sys.modules["pydub.silence"] = MagicMock()

from highrize import HighRize, CompressedClient
from highrize.models import Modality

# --- Mock AI Clients ---

class MockOpenAIResponse:
    def __init__(self, content: str):
        self.choices = [MagicMock(message=MagicMock(content=content))]

class MockOpenAIClient:
    """Simulates the openai.OpenAI client."""
    def __init__(self):
        self.chat = MagicMock()
        self.chat.completions.create = self._mock_create

    def _mock_create(self, messages: List[Dict], **kwargs):
        # In a real simulation, we'd inspect the messages here
        # For now, just return a dummy response
        return MockOpenAIResponse("This is a simulated AI response.")

class MockAnthropicClient:
    """Simulates the anthropic.Anthropic client."""
    def __init__(self):
        self.messages = MagicMock()
        self.messages.create = self._mock_create

    def _mock_create(self, messages: List[Dict], system: str = None, **kwargs):
        return MagicMock(content=[MagicMock(text="Simulated Claude response.")])

# --- Simulation Harness ---

class HighRizeSimulator:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.raw_client = MockOpenAIClient()
        self.compressed_client = CompressedClient(self.raw_client, model=model)
        self.tp = self.compressed_client.tp

    def run_text_simulation(self):
        print("\n[SCENARIO: Long Redundant Prompt]")
        redundant_text = "Please note that I would like to kindly ask you to summarize " * 50
        redundant_text += "\nExample 1: Apple is a fruit.\nExample 2: Banana is a fruit.\nExample 3: Cherry is a fruit.\nExample 4: Date is a fruit."
        
        start = time.perf_counter()
        self.compressed_client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": redundant_text}]
        )
        end = time.perf_counter()
        
        # Get the last result from the report
        res = self.tp.report.results[-1]
        print(f"  Result       : {res}")
        print(f"  Overhead     : {(end - start) * 1000:.2f}ms")

    def run_image_simulation(self):
        print("\n[SCENARIO: High-Res Image (Simulated)]")
        # Generate a fake large base64 string
        fake_b64 = "data:image/jpeg;base64," + "A" * 1000000 # ~1MB string
        
        # Mock ImageCompressor inside TokPress to avoid real PIL calls if needed
        # but since we're using CompressedClient, it will go through tp.compress
        original_compress = self.tp.compress
        self.tp.compress = MagicMock(side_effect=lambda content, modality=None: (
            original_compress(content, modality) if modality != Modality.IMAGE else
            MagicMock(
                original_tokens=1500, # 1024x1024 tile cost
                compressed_tokens=85, # Low detail cost
                savings_pct=94.33,
                modality=Modality.IMAGE,
                original_size_bytes=1000000,
                compressed_size_bytes=50000
            )
        ))

        start = time.perf_counter()
        self.compressed_client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": fake_b64}}
                ]}
            ]
        )
        end = time.perf_counter()
        
        # Manually add the result to the report if we mocked the whole compress function
        # Or better, let's just use the mock result info for the print
        print(f"  Result       : IMAGE: 1500 → 85 tokens, 94.33% saved")
        print(f"  Overhead     : {(end - start) * 1000:.2f}ms")
        
        # Restore
        self.tp.compress = original_compress

    def run_document_simulation(self):
        print("\n[SCENARIO: Long Document (Simulated)]")
        # Simulating a 5000 word document
        doc_text = "Word " * 5000
        
        start = time.perf_counter()
        # We manually call compress or use DocumentCompressor
        from highrize.compressors.document import DocumentCompressor
        dc = DocumentCompressor(token_budget=500, query="Word")
        result = dc.compress(doc_text)
        self.tp.report.add(result, self.tp.cost_per_1k)
        end = time.perf_counter()
        
        print(f"  Result       : {result}")
        print(f"  Overhead     : {(end - start) * 1000:.2f}ms")

    def run_anthropic_simulation(self):
        print("\n[SCENARIO: Anthropic/Claude (Simulated)]")
        anthropic_raw = MockAnthropicClient()
        anthropic_comp = CompressedClient(anthropic_raw, provider="anthropic", model="claude-3-5-sonnet")
        
        # 1x1 transparent GIF
        pixel_b64 = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": "Who are you?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/gif", "data": pixel_b64}}
            ]}
        ]
        
        start = time.perf_counter()
        anthropic_comp.messages.create(
            messages=messages,
            system="Please note that you are a helpful assistant who is very smart."
        )
        end = time.perf_counter()
        
        print(f"  Result       : {anthropic_comp.tp.report.results[-1]}")
        print(f"  Overhead     : {(end - start) * 1000:.2f}ms")
        print("  Full Session :")
        print("    " + anthropic_comp.tp.report.summary().replace("\n", "\n    "))

    def run_video_simulation(self):
        print("\n[SCENARIO: Video Frame Extraction (Simulated)]")
        # Simulating a 100-frame video
        video_path = "test_video.mp4"
        
        # Mock VideoCompressor components
        import cv2
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda x: 100 if x == 7 else (24 if x == 5 else (1024 if x == 3 else 768)) 
        mock_cap.read.return_value = (True, MagicMock())
        cv2.VideoCapture.return_value = mock_cap

        start = time.perf_counter()
        # Video extraction uses ImageCompressor for each frame
        from highrize.compressors.video import VideoCompressor
        vc = VideoCompressor(max_frames=5)
        # Mock ImageCompressor to avoid real Pillow calls
        vc._img_compressor = MagicMock()
        vc._img_compressor.compress.return_value = MagicMock(compressed_tokens=85)
        
        result = vc.compress(video_path)
        self.tp.report.add(result, self.tp.cost_per_1k)
        end = time.perf_counter()
        
        print(f"  Result       : {result}")
        print(f"  Overhead     : {(end - start) * 1000:.2f}ms")

    def show_final_report(self):
        print("\n" + "="*50)
        print("FINAL SIMULATION REPORT")
        print("="*50)
        print(self.tp.report.summary())
        print("="*50)

if __name__ == "__main__":
    print("🚀 Starting HighRize Simulation...")
    sim = HighRizeSimulator()
    
    sim.run_text_simulation()
    sim.run_image_simulation()
    sim.run_document_simulation()
    sim.run_video_simulation()
    sim.run_anthropic_simulation()
    
    sim.show_final_report()
