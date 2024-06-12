import numpy as np  # Importation de la bibliothèque NumPy pour les calculs numériques
import matplotlib.pyplot as plt  # Importation de la bibliothèque Matplotlib pour le tracé des graphiques

def sierpinski_triangle(order):  # Fonction pour générer un triangle de Sierpinski d'un ordre donné
    def sierpinski(order, points):  # Fonction récursive pour calculer les points du triangle de Sierpinski
        if order == 0:  # Condition de base : si l'ordre est 0, retourner les points actuels
            return [points]
        else:  # Sinon, continuer la récursion
            mid = lambda p1, p2: [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]  # Fonction lambda pour calculer le point milieu entre deux points
            top = points[0]  # Point supérieur du triangle
            left = points[1]  # Point gauche du triangle
            right = points[2]  # Point droit du triangle
            tmid = mid(top, left)  # Calculer le point milieu entre le point supérieur et le point gauche
            lmid = mid(left, right)  # Calculer le point milieu entre le point gauche et le point droit
            rmid = mid(right, top)  # Calculer le point milieu entre le point droit et le point supérieur
            return (sierpinski(order-1, [top, tmid, rmid]) +  # Répéter pour le triangle supérieur
                    sierpinski(order-1, [tmid, left, lmid]) +  # Répéter pour le triangle gauche
                    sierpinski(order-1, [rmid, lmid, right]))  # Répéter pour le triangle droit

    points = [[0, 1], [-1, -1], [1, -1]]  # Points initiaux formant un grand triangle
    triangles = sierpinski(order, points)  # Calculer les triangles de Sierpinski jusqu'à l'ordre donné
    for triangle in triangles:  # Pour chaque triangle généré
        plt.fill(*zip(*triangle), 'k')  # Remplir le triangle avec la couleur noire
    plt.gca().set_aspect('equal', adjustable='box')  # Maintenir l'aspect égal des axes
    plt.show()  # Afficher le graphique

def mandelbrot_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon):  # Fonction pour générer l'ensemble de Mandelbrot
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)  # Créer une grille de valeurs en x
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)  # Créer une grille de valeurs en y
    C = X + Y[:, None] * 1j  # Former un tableau de nombres complexes à partir de X et Y
    Z = np.zeros(C.shape, dtype=np.complex64)  # Initialiser Z à zéro pour les itérations
    T = np.zeros(C.shape, dtype=np.float32)  # Initialiser T pour stocker le nombre d'itérations
    for n in range(maxiter):  # Pour chaque itération jusqu'à maxiter
        I = np.less(Z.real*Z.real + Z.imag*Z.imag, horizon)  # Déterminer les indices où Z ne s'échappe pas encore
        Z[I] = Z[I]**2 + C[I]  # Calculer la nouvelle valeur de Z
        T[I] = n  # Stocker le nombre d'itérations
    T = np.log(T + 1)  # Appliquer une échelle logarithmique pour une meilleure représentation visuelle
    plt.imshow(T, extent=[xmin, xmax, ymin, ymax], cmap='hot')  # Afficher l'ensemble avec une colormap chaude
    plt.colorbar()  # Ajouter une barre de couleur pour l'échelle
    plt.show()  # Afficher le graphique

def julia_set(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon, c):  # Fonction pour générer un ensemble de Julia
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)  # Créer une grille de valeurs en x
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)  # Créer une grille de valeurs en y
    Z = X + Y[:, None] * 1j  # Former un tableau de nombres complexes à partir de X et Y
    T = np.zeros(Z.shape, dtype=np.float32)  # Initialiser T pour stocker le nombre d'itérations
    for n in range(maxiter):  # Pour chaque itération jusqu'à maxiter
        I = np.less(Z.real*Z.real + Z.imag*Z.imag, horizon)  # Déterminer les indices où Z ne s'échappe pas encore
        Z[I] = Z[I]**2 + c  # Calculer la nouvelle valeur de Z
        T[I] = n  # Stocker le nombre d'itérations
    T = np.log(T + 1)  # Appliquer une échelle logarithmique pour une meilleure représentation visuelle
    plt.imshow(T, extent=[xmin, xmax, ymin, ymax], cmap='hot')  # Afficher l'ensemble avec une colormap chaude
    plt.colorbar()  # Ajouter une barre de couleur pour l'échelle
    plt.show()  # Afficher le graphique

def koch_snowflake(order, scale=10):  # Fonction pour générer un flocon de Koch
    def koch(order, p1, p2):  # Fonction récursive pour calculer les segments du flocon de Koch
        if order == 0:  # Condition de base : si l'ordre est 0, retourner les points actuels
            return [p1, p2]
        else:  # Sinon, continuer la récursion
            dx = p2[0] - p1[0]  # Calculer la différence en x
            dy = p2[1] - p1[1]  # Calculer la différence en y
            s = np.sqrt(dx*dx + dy*dy) / 3  # Calculer la longueur d'un segment
            a = np.arctan2(dy, dx)  # Calculer l'angle du segment
            p3 = [p1[0] + dx / 3, p1[1] + dy / 3]  # Calculer le premier point tiers
            p5 = [p1[0] + 2 * dx / 3, p1[1] + 2 * dy / 3]  # Calculer le deuxième point tiers
            p4 = [p3[0] + s * np.cos(a - np.pi / 3), p3[1] + s * np.sin(a - np.pi / 3)]  # Calculer le point sommet du triangle
            return (koch(order-1, p1, p3) +  # Répéter pour le premier segment
                    koch(order-1, p3, p4) +  # Répéter pour le deuxième segment
                    koch(order-1, p4, p5) +  # Répéter pour le troisième segment
                    koch(order-1, p5, p2))  # Répéter pour le quatrième segment

    p1 = [0, 0]  # Premier point du triangle initial
    p2 = [scale, 0]  # Deuxième point du triangle initial
    p3 = [scale/2, scale*np.sin(np.pi/3)]  # Troisième point du triangle initial
    points = koch(order, p1, p2)[:-1] + koch(order, p2, p3)[:-1] + koch(order, p3, p1)  # Calculer les points pour chaque segment
    points = np.array(points)  # Convertir les points en tableau NumPy
    plt.plot(points[:,0], points[:,1], 'k')  # Tracer les points
    plt.gca().set_aspect('equal', adjustable='box')  # Maintenir l'aspect égal des axes
    plt.show()  # Afficher le graphique

def burning_ship(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon):  # Fonction pour générer la fractale du Navire en Feu
    X = np.linspace(xmin, xmax, xn, dtype=np.float32)  # Créer une grille de valeurs en x
    Y = np.linspace(ymin, ymax, yn, dtype=np.float32)  # Créer une grille de valeurs en y
    C = X + Y[:, None] * 1j  # Former un tableau de nombres complexes à partir de X et Y
    Z = np.zeros(C.shape, dtype=np.complex64)  # Initialiser Z à zéro pour les itérations
    T = np.zeros(C.shape, dtype=np.float32)  # Initialiser T pour stocker le nombre d'itérations
    for n in range(maxiter):  # Pour chaque itération jusqu'à maxiter
        I = np.less(Z.real*Z.real + Z.imag*Z.imag, horizon)  # Déterminer les indices où Z ne s'échappe pas encore
        Z[I] = (np.abs(Z[I].real) + 1j * np.abs(Z[I].imag))**2 + C[I]  # Calculer la nouvelle valeur de Z en utilisant la valeur absolue des parties réelle et imaginaire
        T[I] = n  # Stocker le nombre d'itérations
    T = np.log(T + 1)  # Appliquer une échelle logarithmique pour une meilleure représentation visuelle
    plt.imshow(T, extent=[xmin, xmax, ymin, ymax], cmap='hot')  # Afficher la fractale avec une colormap chaude
    plt.colorbar()  # Ajouter une barre de couleur pour l'échelle
    plt.show()  # Afficher le graphique
