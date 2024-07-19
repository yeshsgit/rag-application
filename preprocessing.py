import fitz
from progressbar import progressbar
import pandas as pd
import random
from spacy.lang.en import English
import os
import json


class preprocessing():

    def __init__(self, pdf_path, chunk_size, filtered_chunks_file_path, active: bool | None = None) -> None:
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.filtered_chunks_file_path = filtered_chunks_file_path
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        if os.path.exists(filtered_chunks_file_path):
            if active is None or not active:
                print("[INFO] Filtered chunks file found, skipping steps accordingly.\n To overwrite file instantiate preprocessing with active=True")
                self.active = False
                with open(self.filtered_chunks_file_path, 'r') as file:
                    self.filtered_chunk_list = json.load(file)
            else:
                self.active = True
        else:
            self.active = True

    def open_n_read(self) -> list[dict] | None:
        """Opens and creates a list of pages and stats"""
        if self.active:
            doc = fitz.open(self.pdf_path)
            pages_and_text = []

            # Loops through every page and adds info to dict
            # Note the info stored in this list contains additional info like
            # approximate sentence and token counts which can is used for EDA
            print("[INFO] Creating list from from pdf")
            for page_num, page in progressbar(enumerate(doc), max_value=len(doc)):
                text = page.get_text()
                text = self.text_formatter(text)
                string_sentences = self.spacy_sentences(text)
                # -41 page num to keep page numbers consistent with book
                pages_and_text.append({"page number": page_num - 41,
                                       "page char count": len(text),
                                       "page word count": len(text.split(" ")),
                                       "page sentence count est": len(text.split(". ")),
                                       "page sentence count spacy": len(string_sentences),
                                       "page token count": len(text) / 4,
                                       "text": text,
                                       "spacy sentences": string_sentences})
            self.pages_and_text = pages_and_text
            return pages_and_text
        else:
            print("[INFO] Skipping open_n_read function.")

    def text_formatter(self, text: str) -> str:
        return text.replace('\n', ' ').strip()

    def analyse(self, list_name, n: int | None = None) -> list[pd.DataFrame | list[dict]] | pd.DataFrame:
        """Returns stats and specified number of random examples"""
        if not self.active:
            print("[INFO] Note only analysing filtered chunk list.\n For other analysis instantiate preprocessing with active=True")
            list_name = self.filtered_chunk_list
        df = pd.DataFrame(list_name)
        if n is None or n == 0:
            return df.describe()
        elif n < len(list_name) and n > 0:
            samples = random.sample(list_name, k=n)
        else:
            samples = f"""{n} is not a valid number of samples"""
        return df.describe().round(2), samples

    def analyse_page_list(self, n: int | None = None) -> list[pd.DataFrame | list[dict]] | pd.DataFrame:
        analysis = self.analyse(self.pages_and_text, n)
        return analysis

    def analyse_chunk_list(self, n: int | None = None) -> list[pd.DataFrame | list[dict]] | pd.DataFrame:
        analysis = self.analyse(self.pages_and_chunks, n)
        return analysis

    def analyse_filtered_chunk_list(self, n: int | None = None) -> list[pd.DataFrame | list[dict]] | pd.DataFrame:
        analysis = self.analyse(self.filtered_chunk_list, n)
        return analysis

    def spacy_sentences(self, text: str) -> list[str]:
        sentences = list(self.nlp(text).sents)
        string_sentences = [str(sentence) for sentence in sentences]
        return string_sentences

    def get_chunks(self) -> list[dict]:
        """Creates sentence chunks of size specified in config file.
        Adds these chunks to the page list"""
        if self.active:
            print("[INFO] Chunking sentences")
            for item in progressbar(self.pages_and_text):
                item["sentence_chunks"] = self.split_list(item["spacy sentences"])
                item["num_chunks"] = len(item["sentence_chunks"])
            return self.pages_and_text
        else:
            print("[INFO] Skipping get chunks")

    def split_list(self, input_list: list[str]) -> list[list[str]]:
        return [input_list[i:i + self.chunk_size] for i in range(0, len(input_list), self.chunk_size)]

    def create_chunk_list(self) -> list[dict]:
        """Creates list of dicts with chunks and additonal info"""
        if self.active:
            pages_and_chunks = []
            print("[INFO] Creating chunk list")
            for item in progressbar(self.pages_and_text):
                for sentence_chunk in item["sentence_chunks"]:
                    chunk_dict = {}
                    chunk_dict["page number"] = item["page number"]
                    sentence_chunk_combined = " ".join(sentence_chunk).replace("  ", " ")
                    chunk_dict["sentence chunk"] = sentence_chunk_combined
                    chunk_dict["chunk word count"] = len(sentence_chunk_combined.split(" "))
                    chunk_dict["chunk token count"] = len(sentence_chunk_combined) / 4

                    pages_and_chunks.append(chunk_dict)
            self.pages_and_chunks = pages_and_chunks
            return pages_and_chunks
        else:
            print("[INFO] Skipping creating chunk list.")

    def filter_chunk_list(self, min_tokens: int) -> list[dict]:
        """Filters chunks that have less than min_tokens and saves file"""
        if self.active:
            print(f"[INFO] Filtering chunks with less than {min_tokens} tokens..")
            filtered_chunk_list = []
            dropped_chunks = []
            for chunk in progressbar(self.pages_and_chunks):
                if chunk["chunk token count"] > min_tokens:
                    filtered_chunk_list.append(chunk)
                else:
                    dropped_chunks.append({
                        "chunk token count": chunk["chunk token count"],
                        "chunk text": chunk["sentence chunk"]
                    })
            self.filtered_chunk_list = filtered_chunk_list
            with open(self.filtered_chunks_file_path, 'w') as file:
                json.dump(self.filtered_chunk_list, file)
            print(f"[INFO] Successfully saved filtered chunks at {self.filtered_chunks_file_path}")
            self.dropped_chunks = dropped_chunks
            return filtered_chunk_list
        else:
            print("[INFO] Skipping filtering chunks")

    def create_raw_chunk_list(self) -> list:
        """Outputs list of only chunk text with no extra info"""
        print("[INFO] Creating list of raw chunks")
        raw_chunk_list = []
        for chunk in progressbar(self.filtered_chunk_list):
            raw_chunk_list.append(chunk["sentence chunk"] + f" PAGE NUMBER: {chunk['page number']}")
        self.raw_chunk_list = raw_chunk_list
        return raw_chunk_list
