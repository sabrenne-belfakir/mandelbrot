from fractales import sierpinski_triangle, mandelbrot_set, julia_set, koch_snowflake, burning_ship

# Example usage:
if __name__ == "__main__":
    print("Generating Sierpiński Triangle...")
    sierpinski_triangle(order=5)

    print("Generating Mandelbrot Set...")
    mandelbrot_set(xmin=-2.5, xmax=1.5, ymin=-2.0, ymax=2.0, xn=800, yn=800, maxiter=256, horizon=1000.0)

    print("Generating Julia Set...")
    julia_set(xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, xn=800, yn=800, maxiter=256, horizon=1000.0, c=-0.4 + 0.6j)

    print("Generating Koch Snowflake...")
    koch_snowflake(order=4)

    print("Generating Burning Ship...")
    burning_ship(xmin=-2.0, xmax=1.5, ymin=-2.0, ymax=2.0, xn=800, yn=800, maxiter=256, horizon=1000.0)
