# Mandelbrot

## Introduction
Fractals are fascinating mathematical objects that exhibit self-similarity and infinite complexity. Their main characteristic is self-similarity, meaning that when zooming in on a part of a fractal, one observes a structure that resembles the entire set. This property repeats at different scales, giving fractals remarkable beauty and infinite complexity. Fractals are used in various fields, including physics, biology, and computer science, to model complex, natural phenomena. Among the most well-known examples are the Sierpiński triangle, the Mandelbrot set, the Julia set, the Koch snowflake, and the Burning Ship fractal.

## Sierpiński Triangle
The Sierpiński triangle is a self-similar fractal generated using a simple recursive method. It starts with an equilateral triangle. In each iteration, the middle triangle is removed, revealing a repeating pattern that continues infinitely. This fractal is an excellent example of how simple rules can lead to complex and visually captivating structures.

- To generate the Sierpiński triangle:
1. Start with an equilateral triangle.
  2. Divide it into four smaller equilateral triangles by connecting the midpoints of each side.
  3. Remove the central triangle.
  4. Repeat the process for the three remaining triangles.

This iterative process results in a fractal pattern where each smaller triangle exhibits the same structure as the original.

## Mandelbrot Set (Multibrot)
The Mandelbrot set is one of the most famous fractals and is generated using complex numbers. Each point in the Mandelbrot set is determined by iterating the function z = z^2 + c, where z and c are complex numbers. Points that do not escape to infinity under iteration belong to the set. This fractal is notable for its intricate boundary and the infinite complexity revealed at every scale.

- To generate the Mandelbrot set:
1. Define a region in the complex plane, typically ranging from -2.5 to 1.5 on the real axis and -2.0 to 2.0 on the imaginary axis.
  2. For each point c in this region, initialize z to 0.
  3. Iterate the function z = z^2 + c.
  4. Track the number of iterations required for |z| to exceed a certain threshold (indicating divergence).
  5. Points that do not diverge after a specified number of iterations are part of the Mandelbrot set.

## Julia Set
Julia sets are closely related to the Mandelbrot set. They are generated by iterating the function z = z^2 + c for a fixed complex number c. Different values of c produce different Julia sets, each with unique and intricate patterns. Julia sets are interesting because small changes in c can result in vastly different fractal structures.

- To generate a Julia set:
1. Choose a constant complex number c.
  2. Define a grid of points in the complex plane.
  3. For each point z in this grid, iterate the function z = z^2 + c.
  4. Track the number of iterations required for |z| to exceed a certain threshold.
  5. Plot the points based on the iteration count to visualize the Julia set.

## Koch Snowflake
The Koch snowflake is another well-known fractal, generated using an iterative method. Starting with an equilateral triangle, each iteration replaces the middle third of each line segment with two line segments that form an equilateral "bump". This process results in a snowflake-like shape with infinite perimeter and finite area.

- To generate the Koch snowflake:
1. Start with an equilateral triangle.
  2. Divide each side into three equal segments.
  3. Replace the middle segment with two segments that form an equilateral triangle.
  4. Repeat the process for each side of the resulting shape.

This iterative method produces a fractal with a distinctive snowflake pattern.

## Burning Ship
The Burning Ship fractal is similar to the Mandelbrot set but uses a modified iteration: z = (|Re(z)| + i|Im(z)|)^2 + c. This results in a fractal that resembles a ship in flames, with a unique structure that is different from other well-known fractals.

- To generate the Burning Ship fractal:
1. Define a region in the complex plane.
  2. For each point c in this region, initialize z to 0.
  3. Iterate the function z = (|Re(z)| + i|Im(z)|)^2 + c.
  4. Track the number of iterations required for |z| to exceed a certain threshold.
  5. Points that do not diverge after a specified number of iterations are part of the Burning Ship fractal.

## References
- Wikipedia - Fractale
- Parlons Sciences - Qu’est-ce qu’une fractale ?
- NumPy
- Matplotlib
