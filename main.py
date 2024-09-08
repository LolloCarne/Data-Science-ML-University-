import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import  confusion_matrix, mean_squared_error, precision_score,  r2_score, accuracy_score, recall_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def linear_regression(data):
    # Definizione delle variabili (reading score come input e writing score come target)
    X = data[['reading score']]
    y = data['writing score']

    # Creazione del modello di regressione lineare
    model = LinearRegression()
    model.fit(X, y)

    # Previsione e calcolo dei coefficienti
    y_pred = model.predict(X)
    coefficients = model.coef_
    intercept = model.intercept_
    print(f"Coefficiente: {coefficients[0]:.4f}")
    print(f"Intercetta: {intercept:.4f}")

    # Calcolo di R^2 e MSE
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)

    # Plot dei dati e della retta di regressione
    plt.scatter(X, y, color='blue')
    plt.plot(X, y_pred, color='red', linewidth=2)
    plt.title(f'Regressione Lineare: Lettura vs Scrittura\nR^2: {r2:.2f}, MSE: {mse:.2f}')
    plt.xlabel('Reading Score')
    plt.ylabel('Writing Score')
    #plt.show()

    # Analisi dei residui
    residuals = y - y_pred
    plt.hist(residuals, bins=20)
    plt.title('Distribuzione dei residui')
    #plt.show()

    return r2, mse

def EDA(data):
    # Esaminare le prime righe del dataset per comprendere la struttura
    data.head(), data.info()
    # Statistiche descrittive per i punteggi dei test per individuare possibili valori anomali
    print(data[['math score', 'reading score', 'writing score']].describe())

    # Matrice di correlazione per i punteggi dei test
    corr_matrix = data[['math score', 'reading score', 'writing score']].corr()

    # Heatmap della matrice di correlazione
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix for Test Scores')
    #plt.show()

    # Distribuzione univariata dei punteggi dei test
    plt.figure(figsize=(14,6))

    # Istogramma per il punteggio di matematica
    plt.subplot(1, 3, 1)
    sns.histplot(data['math score'], bins=20, kde=True, color='blue')
    plt.title('Distribution of Math Scores')

    # Istogramma per il punteggio di lettura
    plt.subplot(1, 3, 2)
    sns.histplot(data['reading score'], bins=20, kde=True, color='green')
    plt.title('Distribution of Reading Scores')

    # Istogramma per il punteggio di scrittura
    plt.subplot(1, 3, 3)
    sns.histplot(data['writing score'], bins=20, kde=True, color='red')
    plt.title('Distribution of Writing Scores')

    plt.tight_layout()
    #plt.show()

def logistic_regresion(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
    
    # Definizione del modello di regressione logistica
    logistic_model = LogisticRegression()

    # Addestramento del modello
    logistic_model.fit(X_train, y_train)

    # Previsioni sul validation set
    y_test_pred = logistic_model.predict(X_test)

    # Valutazione delle performance del modello con metriche e matrice di confusione
    MR, accuracy = metrics(y_test,y_test_pred)
    print(f"Frequenza di Misclassificazione (MR): {MR * 100:.2f}%")
    print(f"Accuratezza: {accuracy * 100:.2f}%")

    precision,sens = confusionMatrix(y_test,y_test_pred)
    print(f"Precisione: {precision * 100:.2f}%")
    print(f"Sensibilità: {sens * 100:.2f}%")

    # Visualizzazione della decision boundary 
    xx, yy = np.meshgrid(np.linspace(X['reading score'].min(), X['reading score'].max(), 100),
                        np.linspace(X['writing score'].min(), X['writing score'].max(), 100))
    input_pred = pd.DataFrame({'reading score': xx.ravel(), 'writing score': yy.ravel()})
    Z = logistic_model.predict(input_pred)
    Z = Z.reshape(xx.shape)

    # Plot dei punti e del contorno
    plt.figure(figsize=(8,6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    plt.scatter(X_test['reading score'], X_test['writing score'], c=y_test, cmap=plt.cm.coolwarm, edgecolors='k')
    plt.title("Logistic Regression Classification - Decision Boundary")
    plt.xlabel("Reading Score")
    plt.ylabel("Writing Score")
    #plt.show()

    return MR,accuracy,precision,sens

def SVM_CV_TRAIN(kernel, X_train, y_train, X_val, y_val):

    if kernel == 'linear':
        param_grid = {'C': [0.1, 1, 10, 100]}
    elif kernel == 'rbf':
        param_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['scale', 'auto', 0.1, 1]}
    elif kernel == 'poly':
        param_grid = {'C': [0.1, 1, 10, 50, 100], 'degree': [2, 3], 'gamma': ['scale', 'auto']}

    # Definizione del modello SVM
    svm_model = SVC(kernel=kernel)

    # Impostazione del GridSearchCV
    grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy')

    # Addestramento del modello e ricerca dei migliori iperparametri
    grid_search.fit(X_train, y_train)

    # Miglior modello ottenuto da GridSearch
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Stampa dei migliori iperparametri trovati
    print("Migliori iperparametri trovati:", best_params)
    print("Miglior score di validazione:", grid_search.best_score_)

    # Previsioni sul validation set con il miglior modello trovato
    y_val_pred = best_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)

    return best_model, best_params, val_accuracy

def SVM_CV_TEST(X,y):
    # Dividiamo il dataset in training, validation e test set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=100)
    X_val, X_test, y_val, y_test = train_test_split(X_temp,y_temp, test_size=0.5, random_state=100)

    # Creazione dello scaler
    scaler = StandardScaler()
    # Adattamento e trasformazione delle variabili di input
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #Training dei modelli SVM con kernel diversi
    kernels = ['linear', 'rbf', 'poly']
    results = {}
    for kernel in kernels:
        print(f"\nTraining SVM with {kernel} kernel:")
        best_model, best_params, val_accuracy = SVM_CV_TRAIN(kernel, X_train_scaled, y_train, X_val_scaled, y_val)
        results[kernel] = {
            'model': best_model,
            'params': best_params,
            'val_accuracy': val_accuracy
        }

    # Seleziona il miglior modello basato sull'accuratezza di validazione
    best_kernel = ''
    best_accuracy = 0

    for kernel, result in results.items():
        if result['val_accuracy'] > best_accuracy:
            best_accuracy = result['val_accuracy']
            best_kernel = kernel

    best_overall_model = results[best_kernel]['model']
    best_overall_params = results[best_kernel]['params']

    print(f"\nMiglior modello complessivo: SVM con kernel {best_kernel}")
    print(f"Migliori parametri complessivi: {best_overall_params}")

    #valutazione del modello tramite le metriche 
    y_test_pred = best_overall_model.predict(X_test_scaled)
    MR, accuracy = metrics(y_test, y_test_pred)
    print(f"Frequenza di Misclassificazione (MR): {MR * 100:.2f}%")
    print(f"Accuratezza: {accuracy * 100:.2f}%")

    precision,sens = confusionMatrix(y_test,y_test_pred)
    print(f"Precisione: {precision * 100:.2f}%")
    print(f"Sensibilità: {sens * 100:.2f}%")


    return  MR, accuracy, precision,sens
    

def metrics(y_test, y_test_pred):
    # Frequenza di misclassificazione (Misclassification Rate, MR)
    MR = np.mean(y_test_pred != y_test)
    # Accuratezza (Accuratezza = 1 - MR)
    accuracy = 1 - MR

    return MR, accuracy


def confusionMatrix(y_test,y_test_pred):
    cm = confusion_matrix(y_test, y_test_pred)
    #Visualizzazione della Confusion Matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    precision = precision_score(y_test, y_test_pred)
    sens = recall_score(y_test, y_test_pred)

    return precision,sens

def run_experiments_logistic(X, y, n_runs=10):
    MRS, accuracies, precisions, sensitivities = [], [], [], []

    for _ in range(n_runs):
        # Esegui logistic regression
        MR, accuracy, precision, sensitivity = logistic_regresion(X,y)

        accuracies.append(accuracy)
        precisions.append(precision)
        sensitivities.append(sensitivity)
        MRS.append(MR)

    return MRS, accuracies, precisions, sensitivities

def run_experiments_linear(data, n_runs=10):
    r2_scores, mse_scores = [], []
    
    for _ in range(n_runs):
        # Linear regression
        r2_score,mean_squared_error=linear_regression(data)
        r2_scores.append(r2_score)
        mse_scores.append(mean_squared_error)

    return r2_scores, mse_scores

def run_experiments_svm(X, y, n_runs=10):
    MRS, accuracies, precisions, sensitivities = [], [], [], []

    for _ in range(n_runs):
        # Esegui SVM con cross-validation
        MR, accuracy, precision,sens = SVM_CV_TEST(X,y)
        
        accuracies.append(accuracy)
        MRS.append(MR)
        precisions.append(precision)
        sensitivities.append(sens)

    return MRS, accuracies, precisions, sensitivities

def analyze_results(results):
    # Statistiche descrittive (media e deviaz stndrd)
    means = np.mean(results, axis=0)
    stds = np.std(results, axis=0)
    
    # Intervallo di confidenza (95%)
    conf_intervals = stats.t.interval(0.95, len(results)-1, loc=means, scale=stats.sem(results))

    # Plot delle distribuzioni
    plt.figure(figsize=(10, 6))
    sns.histplot(results, bins=30, kde=True)  # Aumenta i bin
    plt.title(f'Distribuzione dei Risultati (n={len(results)})')
    plt.show()

    plt.boxplot(results)
    plt.title('Boxplot dei Risultati')
    plt.show()

    return {"means":means,
            "stds":stds,
            "conf_intervals":conf_intervals}

def show_statistic(X,y,data):

    # Regressione Lineare
    r2_scores, mse_scores = run_experiments_linear(data, n_runs=10)
    print("\nStatistiche Linear Regression (media,deviazione standard,intervallo di confidenza):")   
    final= analyze_results(mse_scores)
    print(f"\nMSE: {final}")
    final= analyze_results(r2_scores)
    print(f"R²: {final}")


    #SVM con best kernel
    MRS, accuracies, precisions, sensitivities = run_experiments_svm(X, y, n_runs=10)
    print("\n Statistiche SVM (Best Kernel):")
    print("\nAccuratezza: ", analyze_results(accuracies))
    print("\nMisclassification Rate: ", analyze_results(MRS))
    print("\nPrecisione: ", analyze_results(precisions))
    print("\nSensibilità: ", analyze_results(sensitivities))
    


    #Regressione logistica
    MRS, logistic_accuracies, logistic_precisions, logistic_sensitivities = run_experiments_logistic(X, y, n_runs=10)
    print("\nStatistiche Logistic Regression:")
    print("\nAccuratezza: ", analyze_results(logistic_accuracies))
    print("\nMisclassification Rate: ", analyze_results(MRS))
    print("\nPrecisione: ", analyze_results(logistic_precisions))
    print("\nSensibilità: ", analyze_results(logistic_sensitivities))


def main():
    # Caricamento dei dati
    data = pd.read_csv('StudentsPerformance.csv')

    #EDA(data)
    linear_regression(data)
    # Set style for plots
    sns.set_style("whitegrid")

    # Creazione della variabile target binaria (punteggio di matematica alto o basso)
    data['math score class'] = (data['math score'] >= data['math score'].mean()).astype(int)

    # Selezione delle variabili di input (caratteristiche) e della variabile target
    X = data[['reading score', 'writing score']]  # Utilizziamo le altre due materie come caratteristiche
    y = data['math score class']  # Target binario

    logistic_regresion(X,y)
    SVM_CV_TEST(X,y)
    #show_statistic(X,y,data)

if __name__ == "__main__":
    main()