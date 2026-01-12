from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_pdf(path):
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, height - 72, "A Brief History of Artificial Intelligence")
    
    c.setFont("Helvetica", 12)
    text = """
    Artificial Intelligence (AI) was founded as an academic discipline in 1956.
    The field was founded on the assumption that human intelligence "can be so precisely described that a machine can be made to simulate it".
    
    In the first decades of the 21st century, highly mathematical-statistical machine learning has dominated the field.
    This technique has proved highly successful, helping to solve many challenging problems throughout industry and academia.
    
    Deep Blue became the first computer chess-playing system to beat a reigning world chess champion, Garry Kasparov, on 11 May 1997.
    
    In 2011, a Jeopardy! quiz show exhibition match, IBM's question answering system, Watson, defeated the two greatest Jeopardy! champions, Brad Rutter and Ken Jennings, by a significant margin.
    
    ChatGPT, launched by OpenAI in November 2022, is based on the GPT-3.5 and later GPT-4 families of large language models (LLMs).
    """
    
    y = height - 100
    for line in text.strip().split('\n'):
        c.drawString(72, y, line.strip())
        y -= 20
        
    c.save()
    print(f"Created PDF at {path}")

if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    create_pdf("sample_data/ai_history.pdf")
