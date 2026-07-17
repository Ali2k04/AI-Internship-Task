import os

while True:

    print("\n==============================")
    print(" Language Detection System")
    print("==============================")
    print("1. Train Model")
    print("2. Predict Language")
    print("3. Batch Prediction")
    print("4. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":
        os.system("python train.py")

    elif choice == "2":
        os.system("python predict.py")

    elif choice == "3":
        os.system("python batch_predict.py")

    elif choice == "4":
        print("Thank You!")
        break

    else:
        print("Invalid Choice")