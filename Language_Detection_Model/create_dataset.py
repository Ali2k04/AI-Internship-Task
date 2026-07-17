import pandas as pd

english = [
    "How are you?",
    "Good morning",
    "Good evening",
    "Good night",
    "Thank you",
    "You're welcome",
    "See you later",
    "Nice to meet you",
    "Have a nice day",
    "What is your name?",
    "My name is John.",
    "Where do you live?",
    "I live in Pakistan.",
    "I love programming.",
    "Python is easy.",
    "Machine learning is fun.",
    "Artificial Intelligence is amazing.",
    "Open the door.",
    "Close the window.",
    "Can you help me?"
]

urdu = [
    "آپ کیسے ہیں؟",
    "صبح بخیر",
    "شام بخیر",
    "شب بخیر",
    "شکریہ",
    "خوش آمدید",
    "پھر ملیں گے",
    "آپ سے مل کر خوشی ہوئی۔",
    "آپ کا دن اچھا گزرے۔",
    "آپ کا نام کیا ہے؟",
    "میرا نام علی ہے۔",
    "آپ کہاں رہتے ہیں؟",
    "میں پاکستان میں رہتا ہوں۔",
    "مجھے پروگرامنگ پسند ہے۔",
    "پائتھن آسان ہے۔",
    "مشین لرننگ دلچسپ ہے۔",
    "مصنوعی ذہانت حیرت انگیز ہے۔",
    "دروازہ کھولو۔",
    "کھڑکی بند کرو۔",
    "کیا آپ میری مدد کر سکتے ہیں؟"
]

spanish = [
    "¿Cómo estás?",
    "Buenos días",
    "Buenas tardes",
    "Buenas noches",
    "Gracias",
    "De nada",
    "Hasta luego",
    "Mucho gusto",
    "Que tengas un buen día",
    "¿Cómo te llamas?",
    "Me llamo Juan.",
    "¿Dónde vives?",
    "Vivo en Pakistán.",
    "Me encanta programar.",
    "Python es fácil.",
    "El aprendizaje automático es divertido.",
    "La inteligencia artificial es increíble.",
    "Abre la puerta.",
    "Cierra la ventana.",
    "¿Puedes ayudarme?"
]

french = [
    "Comment ça va ?",
    "Bonjour",
    "Bonsoir",
    "Bonne nuit",
    "Merci",
    "Je vous en prie.",
    "À bientôt.",
    "Enchanté de vous rencontrer.",
    "Passez une bonne journée.",
    "Comment vous appelez-vous ?",
    "Je m'appelle Jean.",
    "Où habitez-vous ?",
    "J'habite au Pakistan.",
    "J'aime programmer.",
    "Python est facile.",
    "L'apprentissage automatique est amusant.",
    "L'intelligence artificielle est incroyable.",
    "Ouvrez la porte.",
    "Fermez la fenêtre.",
    "Pouvez-vous m'aider ?"
]

german = [
    "Wie geht es dir?",
    "Guten Morgen",
    "Guten Abend",
    "Gute Nacht",
    "Danke",
    "Bitte schön.",
    "Bis später.",
    "Freut mich dich kennenzulernen.",
    "Schönen Tag noch.",
    "Wie heißt du?",
    "Ich heiße Hans.",
    "Wo wohnst du?",
    "Ich wohne in Pakistan.",
    "Ich programmiere gern.",
    "Python ist einfach.",
    "Maschinelles Lernen macht Spaß.",
    "Künstliche Intelligenz ist erstaunlich.",
    "Öffne die Tür.",
    "Schließe das Fenster.",
    "Kannst du mir helfen?"
]

rows = []

for _ in range(10):  # 20 × 10 = 200 rows
    for sentence in english:
        rows.append([sentence, "English"])
    for sentence in urdu:
        rows.append([sentence, "Urdu"])
    for sentence in spanish:
        rows.append([sentence, "Spanish"])
    for sentence in french:
        rows.append([sentence, "French"])
    for sentence in german:
        rows.append([sentence, "German"])

df = pd.DataFrame(rows, columns=["text", "language"])

df.to_csv("dataset.csv", index=False, encoding="utf-8")

print("dataset.csv created successfully!")