import random
import codecs
import os

# 1.0 - random number generator
random.seed(123)


class DataProcessing:

    def __init__(self, input_file):
        self.input_file = input_file

    # DELETE ALL FILES
    def delete_all_files(self):
        files_in_directory = os.listdir()

        # Filter the list to include only those named 'paragraph_x.txt'
        filtered_files = [file for file in files_in_directory if "paragraph_" in file and file.endswith(".txt")]

        # Delete each file
        for file in filtered_files:
            os.remove(file)

        print(f"Removed: {len(files_in_directory)}")

    # partition file into seperate paragraphs. Each paragraph will be a seperate document.
    def partition(self, input_file):
        # open and load utf-8 encoded file
        with codecs.open(input_file, "r", "utf-8") as file:
            lines = file.readlines()

            chunks = []
            paragraph = []
            for line in lines:
                line = line.strip()
                if line:  # if the line is not empty, add to current paragraph
                    paragraph.append(line)
                else:  # if the line is empty and there's an existing paragraph, end the current paragraph
                    if paragraph:
                        chunks.append(' '.join(paragraph))
                        paragraph = []
            # Add the last paragraph if the file doesn't end with an empty line
            if paragraph:
                chunks.append(' '.join(paragraph))

        for index, paragraph in enumerate(chunks, start=1):
            output_filename = f"paragraph_{index}.txt"
            with codecs.open(output_filename, 'w', "utf-8") as out_file:
                out_file.write(paragraph)

        print(f"Partitioned {len(chunks)} into seperate files.")

    # 1.3 - remove paragraphs containing the word "Gutenberg" (= headers and footers)

    def filter_files(self, directory_path, target_word):
        files_in_directory = os.listdir()

        # Filter out directories, we only want files
        files_only = [f for f in files_in_directory if os.path.isfile(os.path.join(directory_path, f))]

        for filename in files_only:
            with open(os.path.join(directory_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                if target_word in content:
                    os.remove(os.path.join(directory_path, filename))
                    print(f"Removed: {filename}")

        # Filter the list to include only those named 'paragraph_x.txt'
        filtered_files = [file for file in files_in_directory if "paragraph_" in file and file.endswith(".txt")]

        file_count = 0
        for file in filtered_files:
            if word in file:
                os.remove(file)
                file_count += 1
        # Delete each file

        print(f"Removed {file_count} files containing {word}")


def main():
    input_filename = "pg3300.txt"
    target_word = "Gutenberg"
    directory_path = "../"
    processor = DataProcessing(input_filename)

    # Remove all generated files
    # processor.delete_all_files()

    # Partition all paragraphs into documents
    # processor.partition(input_filename)

    # Remove all paragraphs containting a certain word
    processor.filter_files(directory_path, target_word)


if __name__ == "__main__":
    main()
