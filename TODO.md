# TODO: Fix Output Formatting and Citations Display

- [x] Modify rag.py's generate_answer method to return a dict with "answer" and "citations" keys
- [x] Add checks in rag.py for missing metadata keys (case_name, section) and use fallbacks
- [x] Update app.py to handle the new return format from rag.generate_answer
- [x] Update app.py to display answer and citations in separate, clearly labeled sections
- [x] Format citations in app.py as bullet points for better readability
- [ ] Test the UI to confirm citations display properly and output formatting is improved
