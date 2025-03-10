import nbformat
from nbclient import NotebookClient
import logging

class Notebook:
    def __init__(self, kernel_name="python3", timeout=10, cwd=None):
        self.notebook = nbformat.v4.new_notebook()
        self.kernel_name = kernel_name
        self.timeout = timeout
        self.cwd = cwd
        
    def add_block(self, code):
        cell = nbformat.v4.new_code_cell(source=code)
        self.notebook.cells.append(cell)

    def edit_block(self, index, new_content):
        if 0 <= index < len(self.notebook.cells):
            cell = self.notebook.cells[index]
            old_content = cell["source"]
            cell["source"] = new_content
        else:
            logging.error(f"Index {index} is out of range. No block edited.")
            raise IndexError(f"Index {index} is out of range. No block edited.")
        
    def delete_block(self, index):
        if 0 <= index < len(self.notebook.cells):
            deleted_cell = self.notebook.cells.pop(index)
        else:
            logging.error(f"Index {index} is out of range. No block deleted.")

    def execute_block(self, index):
        status = True
        result = ""
        if index < 0 or index >= len(self.notebook.cells):
            raise IndexError(f"Index {index} is out of range. No block executed.")
    
        client = NotebookClient(
            self.notebook,
            kernel_name=self.kernel_name,
            timeout=self.timeout,
            allow_errors=True
        )
        if self.cwd:
            client.cwd = self.cwd

        try:
            client.execute(cwd=self.cwd)
        except Exception as e:
            return f"Execution failed: {e}", False
        cell = self.notebook.cells[index]
        
        outputs = cell.get("outputs", [])
        for output in outputs:
            output_type = output.get("output_type")
            if output_type == "stream":
                result += output.get("text", "")
            elif output_type == "execute_result":
                data = output.get("data", {})
                result += data.get("text/plain", "")
            elif output_type == "display_data":
                data = output.get("data", {})
                if "text/plain" in data:
                    result += data["text/plain"]
                elif "text/html" in data:
                    result += f"[HTML Output]: {data['text/html']}\n"
            elif output_type == "error": 
                status = False
                ename = output.get("ename", "Unknown Error")
                evalue = output.get("evalue", "No error message")
                traceback = self._clean_traceback(output.get("traceback", []))
                result += f"{ename}: {evalue}\n{traceback}\n"
        return result, status
    
    def save_notebook(self, output_path):
        with open(output_path, "w", encoding="utf-8") as f:
            nbformat.write(self.notebook, f)
        logging.info(f"Notebook saved to {output_path}.")
        
    def execute_last_block(self):
        if not self.notebook.cells:
            return "No blocks to execute.", False
        client = NotebookClient(
            self.notebook,
            kernel_name=self.kernel_name,
            timeout=self.timeout,
            allow_errors=True
        )
       
        try:
            client.execute(cwd=self.cwd)
        except Exception as e:
            return str(e), False
        
        last_cell = self.notebook.cells[-1]
        outputs = last_cell.get("outputs", [])
        result = ""
        status = True
        for output in outputs:
            output_type = output.get("output_type")
            if output_type == "stream":
                result += output.get("text", "")
            elif output_type == "execute_result":
                data = output.get("data", {})
                result += data.get("text/plain", "")
            elif output_type == "error":
                status = False
                ename = output.get("ename", "Unknown Error")
                evalue = output.get("evalue", "No error message")
                traceback = self._clean_traceback(output.get("traceback", []))
                result += f"{ename}: {evalue}\n{traceback}\n"
                
        return result.strip(), status
    
    def load_notebook(self, input_path):
        """
        Load a notebook from a file and replace the current notebook content.
        """
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                self.notebook = nbformat.read(f, as_version=4)
            logging.info(f"Notebook loaded from {input_path}.")
        except Exception as e:
            logging.error(f"Failed to load notebook from {input_path}: {e}")
            raise
        
    def add_or_replace_block(self, code):
        if self.notebook.cells:
            _, last_status = self.execute_last_block()
            if not last_status:
                self.notebook.cells[-1]["source"] = code
            else:
                self.notebook.cells.append(nbformat.v4.new_code_cell(source=code))
               
        else:
            self.notebook.cells.append(nbformat.v4.new_code_cell(source=code))
     

        return self.execute_last_block()

    def merge_all_blocks(self):
        merged_code = "\n\n".join(cell["source"] for cell in self.notebook.cells if cell["cell_type"] == "code")
        return merged_code
    

    
    def _clean_traceback(self, traceback_lines):
        cleaned_lines = []
        for line in traceback_lines:
            if "\x1b" in line or "Traceback" in line and not line.strip():
                continue
            cleaned_lines.append(line.strip())
        return "\n".join(cleaned_lines)
    
    def get_all_execution_outputs(self):
        """
        Execute all blocks in the notebook and return their code and outputs as a list of tuples (block code, execute result).
        """
        client = NotebookClient(
            self.notebook,
            kernel_name=self.kernel_name,
            timeout=self.timeout,
            allow_errors=True
        )
        if self.cwd:
            client.cwd = self.cwd

        try:
            client.execute(cwd=self.cwd)
        except Exception as e:
            logging.error(f"Execution failed: {e}")
            return [(cell["source"], f"Execution failed: {e}") for cell in self.notebook.cells if cell["cell_type"] == "code"]

        results = []
        for cell in self.notebook.cells:
            if cell["cell_type"] == "code":
                code = cell["source"]
                result = ""
                outputs = cell.get("outputs", [])
                for output in outputs:
                    output_type = output.get("output_type")
                    if output_type == "stream":
                        result += output.get("text", "")
                    elif output_type == "execute_result":
                        data = output.get("data", {})
                        result += data.get("text/plain", "")
                    elif output_type == "error":
                        ename = output.get("ename", "Unknown Error")
                        evalue = output.get("evalue", "No error message")
                        traceback = self._clean_traceback(output.get("traceback", []))
                        result += f"{ename}: {evalue}\n{traceback}\n"
                results.append((code.strip(), result.strip()))
        return results 