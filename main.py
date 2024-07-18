import config
import get_pdf
import preprocessing
import embedder
import generation
import gradio as gr


def main():
    pdf_downloader = get_pdf.get_pdf(config.file_path, config.file_url)
    pdf_downloader.download_pdf()

    preprocessor = preprocessing.preprocessing(config.file_path, config.chunk_size, config.filtered_chunks_file_path, active=False)
    preprocessor.open_n_read()
    # print('Analysis:\n', preprocessor.analyse_page_list(1))
    preprocessor.get_chunks()
    # print('Analysis:\n', preprocessor.analyse_page_list(1))
    preprocessor.create_chunk_list()
    # print('Analysis:\n', preprocessor.analyse_chunk_list(1))
    preprocessor.filter_chunk_list(min_tokens=30)
    # print('Analysis:\n', preprocessor.analyse_filtered_chunk_list(5))
    raw_chunk_list = preprocessor.create_raw_chunk_list()

    # filtered chunk list is used to get page numbers for chunks used
    filtered_chunk_list = preprocessor.filtered_chunk_list
    embedding_generator = embedder.embedder(raw_chunk_list, config.embedding_model)
    collection = embedding_generator.embed()

    generator = generation.generation(filtered_chunk_list, collection, embedding_generator, config.llm_model, config.file_name)
    gr.ChatInterface(generator.chat).launch()


if __name__ == "__main__":
    main()
