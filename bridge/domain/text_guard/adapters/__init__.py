"""Text Guard adapters for different file and stream formats.

Each adapter knows how to extract text from a specific format,
run it through the TextGuard engine, and put corrected text back.

Pure adapters in domain layer:
    * text.py: plain text files (.md, .txt, .json, .yaml, .html)
    * streaming.py: token-stream reassembly for live bot output

Application-layer adapters (require third-party libs):
    * docx adapter (python-docx)
    * xlsx adapter (openpyxl)
    * pptx adapter (python-pptx)
    * csv adapter
"""
