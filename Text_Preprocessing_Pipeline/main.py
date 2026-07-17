from preprocessing import TextPreprocessor

processor = TextPreprocessor()

while True:

    print("\n===============================")
    print("TEXT PREPROCESSING PIPELINE")
    print("===============================")

    print("1. Process Single Sentence")
    print("2. Process CSV Dataset")
    print("3. Exit")

    choice = input("\nEnter choice: ")

    if choice == "1":

        text = input("\nEnter sentence:\n")

        result = processor.preprocess(text)

        print("\nFinal Output:")
        print(result)

    elif choice == "2":

        input_file = input("CSV filename: ")

        column = input("Column containing text: ")

        output_file = "processed_dataset.csv"

        processor.process_csv(
            input_file,
            output_file,
            column
        )

    elif choice == "3":

        print("Goodbye!")
        break

    else:
        print("Invalid choice.")