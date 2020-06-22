import pandas as pd
import csv
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


def prepare_data():
    line = 0
    data = []
    matrix_rows = []
    matrix_cols = []
    with open('ratings.csv', 'r') as file:
        datareader = csv.reader(file, delimiter=',')
        for row in datareader:
            if line > 0:
                data.append(float(row[2]))
                matrix_rows.append(int(row[0]))
                matrix_cols.append(int(row[1]))
            line += 1

    return csr_matrix((data, (matrix_rows, matrix_cols)), dtype=float)


def generate_recommendations(x, my_ratings, cols1):
    # normalizujemy macierz
    x_normalized = normalize(x, axis=0)
    # normalizujemy macierz naszych ocen
    my_ratings_normalized = normalize(my_ratings, axis=0)
    # obliczamy podobienstwo cosinusowe z kazdym
    # uzytkownikiem (mnozenie macierzowe)
    z = x_normalized.dot(my_ratings_normalized)

    # normalizujemy otrzymany wektor (reprezentuje nasz profil filmowy)
    z_normalized = normalize(z, axis=0)

    # teraz musimy obliczyc podobienstwo cosinusowe
    # miedzy naszym profilem, a kazda kolumna macierzy
    # aby znalezc takie filmy, ktore sa podobne do naszego
    # profilu - sortujemy po otrzymanym podobienstwie
    # i w ten sposob dostajemy rekomendacje
    x_normalized_transposed = x_normalized.transpose()

    recomendations = x_normalized_transposed.dot(z_normalized)
    recomendations = recomendations.toarray()
    recomendations_flat = [
        item for sublist in recomendations for item in sublist]

    # tworzymy data frame bez zerowego elementu, bo nie ma filmu o id = 0
    data = {"cos(Theta):": recomendations_flat[1:],
            "movies_id:": list(range(1, cols1))
            }
    df = pd.DataFrame(data, columns=["cos(Theta):", "movies_id:"])
    # sortujemy dataframe po wartosci cos
    df = df.sort_values(by="cos(Theta):", ascending=False)
    # print(df.to_string(index=False)) # - mozna wyswietlic sobie
    # dataframe z wartosciami cos(Theta) i movie_id

    # wswietlamy kolejne tytuly pierwszych 100
    # rekomendowanych filmow (bez tych dla ktorych cos(Theta) wyszedl 0)
    movies = pd.read_csv("movies.csv")
    counter = 0
    print("\nRekomendowane filmy: ")
    print("==========================================")
    for movie_id in df["movies_id:"]:
        if counter > 100:
            break
        similarity = df.loc[df["movies_id:"] == movie_id]["cos(Theta):"]
        if similarity.values[0] != 0:
            movie_row = movies.loc[movies["movieId"] == movie_id]["title"]
            if not movie_row.empty and counter <= 100:
                movie_title = str(movie_row.values[0])
                print(movie_title)
        counter += 1


# generujemy macierz ocen
rating_matrix = prepare_data()
(rows, cols) = rating_matrix.shape

# tworzymy wlasna ocene filmow (jako wektor kolumnowy)
# ratings[id_filmu] = ocena
# taki sposob ustawiania wartosci nie jest najbardziej oplacalny,
# ale jesli robimy to tylko kilka razy to nie ma to znaczenia
ratings = csr_matrix((cols, 1), dtype=float)

ratings[2571] = 5  # matrix
ratings[32] = 4  # Twelve Monkeys
ratings[260] = 5  # Star Wars IV
ratings[1097] = 4  # E.T. the Extra-Terrestrial (1982)

# generujemy rekomendacje
generate_recommendations(rating_matrix, ratings, cols)

