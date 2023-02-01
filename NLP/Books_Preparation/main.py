import nltk, re, pandas as pd, random, numpy as np
from nltk.corpus import stopwords
# nltk.download() # -->> uncomment it to download all Nltk Packages.

################################################################################## Functions.
class Text_Data_Partitioning:
    Partitioning_DataFrame1 = pd.DataFrame()    # Create two DataFrames for horizontal and vertical shapes.
    Partitioning_DataFrame2 = pd.DataFrame()

    @staticmethod
    def read_books(book_name):  # for example -->> Input : "austen-emma.txt" # Output -->> book content as string type.
        return nltk.corpus.gutenberg.raw(book_name)

    @staticmethod
    def get_title(book_name):   # for example -->> Input : "austen-emma.txt" # Output -->> book title as string type.
        book = Text_Data_Partitioning.read_books(book_name)
        Text_Data_Partitioning.Book_Title = re.findall('^\[(.*)\]', book)
        Text_Data_Partitioning.Book_Title = Text_Data_Partitioning.Book_Title[0]
        return Text_Data_Partitioning.Book_Title

    @staticmethod
    def book_preprocessing(book_name):     # for example -->> Input : "austen-emma.txt" # Output -->> book content as list of words after book cleaning.
        book = Text_Data_Partitioning.read_books(book_name)
        Text_Data_Partitioning.Book_Title = '[' + Text_Data_Partitioning.Book_Title + ']'
        book = book.replace(Text_Data_Partitioning.Book_Title, '')
        book = re.sub('(CHAPTER(.*))|(Chapter(.*))', '', book)
        book = re.sub('(VOLUME(.*))|(Volume(.*))', '', book)
        book = re.sub('^$\n', '', book, flags=re.MULTILINE)
        book = re.sub('\. *(\W)', '.\n\n', book)
        book = re.sub('[^\w\s]', '', book)
        book = book.lower()
        book = book.split()
        all_eng_words = set(stopwords.words('english'))
        book = [word for word in book if not word in all_eng_words]
        return book

    @staticmethod
    def book_partitioning(book_list):  # for example -->> Input : list of words (you can obtain this from book_preprocessing(book_name) function) # Output -->> book content as list of lists (every 100 word is in a list).
        book_list = [book_list[i:i + 100] for i in range(0, len(book_list), 100)]
        random.shuffle(book_list)
        book_list = book_list[0:200]
        return book_list

    @staticmethod
    def create_horizontal_dataframe(partitions_list, book_title):  # for example -->> Input : list of lists (you can obtain this from book_partitioning(book_list) function) & Book title in str data type (you can obtain this from get_title(book_name) function) # Output -->> book content as a horizontal dataframe.
        Text_Data_Partitioning.Partitioning_DataFrame1[book_title] = pd.Series(partitions_list)
        if len(Text_Data_Partitioning.Partitioning_DataFrame1[book_title]) < 200:
            null = np.empty((200 - len(Text_Data_Partitioning.Partitioning_DataFrame1[book_title]), 1))
            null[:] = pd.Series(np.nan)
            null = null.squeeze()
            null = pd.Series(null)
            Text_Data_Partitioning.Partitioning_DataFrame1 = Text_Data_Partitioning.Partitioning_DataFrame1[book_title].append([null], ignore_index=True)
            Text_Data_Partitioning.Partitioning_DataFrame1 = pd.DataFrame(Text_Data_Partitioning.Partitioning_DataFrame1)
            Text_Data_Partitioning.Partitioning_DataFrame1.columns = [book_title]
        return Text_Data_Partitioning.Partitioning_DataFrame1

    @staticmethod
    def create_vertical_dataframe(partitions_list, book_title):  # for example -->> Input : list of lists (you can obtain this from book_partitioning(book_list) function) & Book title in str data type (you can obtain this from get_title(book_name) function) # Output -->> book content as a vertical dataframe.
        book_names = [book_title] * len(partitions_list)
        temp_df = pd.DataFrame({'Partitions': partitions_list, 'Books Titles': book_names})
        Text_Data_Partitioning.Partitioning_DataFrame2 = Text_Data_Partitioning.Partitioning_DataFrame2.append(temp_df, ignore_index=True)
        Text_Data_Partitioning.Partitioning_DataFrame2 = Text_Data_Partitioning.Partitioning_DataFrame2.sample(frac=1).reset_index(drop=True)
        del temp_df
        return Text_Data_Partitioning.Partitioning_DataFrame2


################################################################################## Main

temp = 0
while int(temp) != int(-1):  # This loop will allow user to append any books from nltk.corpus.gutenberg into either horizontal dataframe or Vertical dataframe
    print("Every time you choose book it will be appended to the DataFrame (Whether you Horizontal DataFrame or Vertical \n")

    List_Of_Books = list(nltk.corpus.gutenberg.fileids())
    print(List_Of_Books)
    User_Choice = input("Pleas choose the desired book (Enter book name with .txt): \n")

    class_instance = Text_Data_Partitioning()
    Book_Title = class_instance.get_title(User_Choice)
    Book = class_instance.book_preprocessing(User_Choice)
    Partitions = class_instance.book_partitioning(Book)

    User_Choice_DF_Shape = input("Pleas choose the desired shape of the DataFrame: Horizontal:1 Vertical:2 \n")
    if int(User_Choice_DF_Shape) == 1:
        DataFrame1 = class_instance.create_horizontal_dataframe(Partitions, Book_Title)
        print(DataFrame1.head())

    elif int(User_Choice_DF_Shape) == 2:
        DataFrame2 = class_instance.create_vertical_dataframe(Partitions, Book_Title)
        print(DataFrame2.head())

    temp = input('Do you want to add another book? Yes:1  NO:-1 : \n')