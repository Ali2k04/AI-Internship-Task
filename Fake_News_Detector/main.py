import os

while True:

    print("\n==============================")
    print(" Fake News Detection System ")
    print("==============================")

    print("1. Train Model")
    print("2. Predict News")
    print("3. Batch Prediction")
    print("4. Feature Importance")
    print("5. Exit")

    choice = input("\nEnter Choice: ")

    if choice == "1":
        os.system("python train_model.py")

    elif choice == "2":
        os.system("python predict.py")

    elif choice == "3":
        os.system("python batch_predict.py")

    elif choice == "4":
        os.system("python feature_importance.py")

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid choice.")