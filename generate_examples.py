#!/usr/bin/env python3
"""
Generate Random Examples for Neural Automata
============================================

This script generates new random examples for both tasks:
1. String images for regex matching
2. Arithmetic expression images

Author: Neural Automata Team
"""

import os
import sys
import random
import string
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.stitch_images import generate_expressions, create_expression

DIGIT_CHOICES = [str(i) for i in range(10)]
OPERATOR_CHOICES = ['+', '-', '%', '*']


class StringImageGenerator:
    """Generate random string images for regex task."""

    def __init__(self, font_size=20):
        self.font_size = font_size
        try:
            # Try to use a common system font
            self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except:
            # Fallback to default font
            self.font = ImageFont.load_default()

    def create_letter_image(self, letter, size=(28, 28)):
        """Create a single letter image."""
        # Create white background
        img = Image.new('L', size, color=255)
        draw = ImageDraw.Draw(img)

        # Get text size to center it
        bbox = draw.textbbox((0, 0), letter, font=self.font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Center the text
        x = (size[0] - text_width) // 2
        y = (size[1] - text_height) // 2

        # Draw black text
        draw.text((x, y), letter, fill=0, font=self.font)

        return np.array(img) / 255.0  # Normalize to [0, 1]

    def generate_random_string(self, length=None, regex_pattern=None):
        """
        Generate a random string of lowercase letters.

        Args:
            length: Length of string (random 3-10 if None)
            regex_pattern: Optional regex pattern to match

        Returns:
            Generated string
        """
        if length is None:
            length = random.randint(3, 10)

        # Generate random lowercase string
        return ''.join(random.choices(string.ascii_lowercase, k=length))

    def string_to_images(self, text):
        """Convert a string to a list of letter images."""
        return [self.create_letter_image(char) for char in text.lower()]

    def save_string_sample(self, text, output_path):
        """Save a string as a series of images."""
        images = self.string_to_images(text)

        # Stack into tensor
        image_tensor = torch.stack([torch.from_numpy(img).float() for img in images])

        # Create output directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save as torch tensor
        torch.save(image_tensor, output_path)

        return image_tensor


class ArithmeticImageGenerator:
    """Generate random arithmetic expression images."""

    def __init__(self, data_path=None):
        """
        Initialize with handwritten digit/operator dataset.

        Args:
            data_path: Path to the handwritten digits and operators dataset
        """
        if data_path is None:
            data_path = 'data/handwritten-digits-and-operators/training.npy'

        if not os.path.exists(data_path):
            print(f"Warning: Dataset not found at {data_path}")
            print("You can download it from:")
            print("https://www.kaggle.com/datasets/michelheusser/handwritten-digits-and-operators/data")
            self.data = None
        else:
            self.data = np.load(data_path, allow_pickle=True)
            self.digits = self.data[np.isin(self.data[:, 1], DIGIT_CHOICES)]
            self.operators = self.data[np.isin(self.data[:, 1], OPERATOR_CHOICES)]

    def generate_random_expression(self, num_operands=2, operand_length=1):
        """
        Generate a random arithmetic expression.

        Args:
            num_operands: Number of operands (2-10)
            operand_length: Number of digits per operand (1-5)

        Returns:
            tuple: (operands, operators) where operands is list of lists
        """
        operands = []
        for _ in range(num_operands):
            operand = [random.choice(DIGIT_CHOICES) for _ in range(operand_length)]
            # Ensure no leading zero
            while operand[0] == '0' and len(operand) > 1:
                operand[0] = random.choice([str(i) for i in range(1, 10)])
            operands.append(operand)

        operators = [random.choice(OPERATOR_CHOICES) for _ in range(num_operands - 1)]

        return operands, operators

    def expression_to_string(self, operands, operators):
        """Convert operands and operators to expression string."""
        result = []
        for i, operand in enumerate(operands):
            result.append(''.join(operand))
            if i < len(operators):
                result.append(operators[i])
        return ''.join(result)

    def evaluate_expression(self, operands, operators):
        """Evaluate the expression left to right."""
        # Convert first operand to number
        result = int(''.join(operands[0]))

        for i, op in enumerate(operators):
            next_operand = int(''.join(operands[i + 1]))

            if op == '+':
                result = result + next_operand
            elif op == '-':
                result = result - next_operand
            elif op == '*':
                result = result * next_operand
            elif op == '%':
                if next_operand == 0:
                    # Avoid division by zero
                    result = 0
                else:
                    result = result / next_operand

        return result

    def create_expression_image(self, operands, operators, output_path=None):
        """Create an image of the arithmetic expression."""
        if self.data is None:
            print("Error: Dataset not loaded. Cannot create expression image.")
            return None

        # Get random images for each digit and operator
        expression_images = []

        for i, operand in enumerate(operands):
            # Add digit images for this operand
            for digit in operand:
                digit_samples = self.digits[self.digits[:, 1] == digit]
                image, label = random.choice(digit_samples.tolist())
                expression_images.append(image)

            # Add operator image if not last operand
            if i < len(operators):
                op = operators[i]
                op_samples = self.operators[self.operators[:, 1] == op]
                image, label = random.choice(op_samples.tolist())
                expression_images.append(image)

        # Stack images horizontally
        stitched = np.hstack(expression_images)

        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as PNG
            img = Image.fromarray((stitched * 255).astype(np.uint8))
            img.save(output_path)

        return stitched

    def create_expression_data(self, operands, operators, output_path=None):
        """Create data array for neural automaton (separate images)."""
        if self.data is None:
            print("Error: Dataset not loaded. Cannot create expression data.")
            return None

        expression_data = []

        for i, operand in enumerate(operands):
            # Add digit images for this operand
            for digit in operand:
                digit_samples = self.digits[self.digits[:, 1] == digit]
                image, label = random.choice(digit_samples.tolist())
                expression_data.append(image)

            # Add operator image if not last operand
            if i < len(operators):
                op = operators[i]
                op_samples = self.operators[self.operators[:, 1] == op]
                image, label = random.choice(op_samples.tolist())
                expression_data.append(image)

        # Stack into numpy array
        data_array = np.stack(expression_data)

        if output_path:
            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save as numpy array
            np.save(output_path, data_array)

        return data_array


def demo_string_generation():
    """Demonstrate generating random string examples."""
    print("\n" + "="*70)
    print("DEMO: Generating Random String Examples")
    print("="*70)

    generator = StringImageGenerator()

    # Generate 5 random strings
    output_dir = 'generated_examples/strings'
    os.makedirs(output_dir, exist_ok=True)

    for i in range(5):
        # Generate random string
        random_string = generator.generate_random_string(length=random.randint(4, 8))

        print(f"\nExample {i+1}: '{random_string}'")

        # Save as tensor
        output_path = os.path.join(output_dir, f'string_{i}.pt')
        images = generator.save_string_sample(random_string, output_path)

        print(f"  Saved to: {output_path}")
        print(f"  Shape: {images.shape}")

        # Visualize
        fig, axes = plt.subplots(1, len(random_string), figsize=(len(random_string) * 1.5, 2))
        if len(random_string) == 1:
            axes = [axes]

        for j, char in enumerate(random_string):
            axes[j].imshow(images[j], cmap='gray')
            axes[j].axis('off')
            axes[j].set_title(char)

        plt.suptitle(f"String: '{random_string}'")
        plt.tight_layout()
        viz_path = os.path.join(output_dir, f'string_{i}_viz.png')
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Visualization: {viz_path}")

    print(f"\nGenerated {5} random string examples in '{output_dir}'")


def demo_arithmetic_generation():
    """Demonstrate generating random arithmetic examples."""
    print("\n" + "="*70)
    print("DEMO: Generating Random Arithmetic Examples")
    print("="*70)

    generator = ArithmeticImageGenerator()

    if generator.data is None:
        print("\nSkipping arithmetic generation (dataset not available)")
        return

    # Generate examples with different complexities
    output_dir = 'generated_examples/arithmetic'
    os.makedirs(output_dir, exist_ok=True)

    examples = [
        (2, 1, "Simple 2 operands"),
        (3, 1, "3 operands"),
        (2, 2, "2-digit operands"),
        (5, 1, "Complex 5 operands"),
    ]

    for i, (num_ops, op_len, desc) in enumerate(examples):
        print(f"\nExample {i+1}: {desc}")

        # Generate random expression
        operands, operators = generator.generate_random_expression(num_ops, op_len)

        expr_str = generator.expression_to_string(operands, operators)
        result = generator.evaluate_expression(operands, operators)

        print(f"  Expression: {expr_str}")
        print(f"  Result: {result}")

        # Create data array (for neural automaton)
        data_path = os.path.join(output_dir, f'expr_{i}_data.npy')
        data = generator.create_expression_data(operands, operators, data_path)
        print(f"  Data saved to: {data_path}")
        print(f"  Shape: {data.shape}")

        # Create stitched image (for VLM)
        img_path = os.path.join(output_dir, f'expr_{i}_image.png')
        img = generator.create_expression_image(operands, operators, img_path)
        print(f"  Image saved to: {img_path}")

    print(f"\nGenerated {len(examples)} arithmetic examples in '{output_dir}'")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("RANDOM EXAMPLE GENERATION")
    print("="*70)

    # Demo 1: String generation
    demo_string_generation()

    # Demo 2: Arithmetic generation
    demo_arithmetic_generation()

    print("\n" + "="*70)
    print("GENERATION COMPLETE!")
    print("="*70)
    print("\nGenerated examples are in the 'generated_examples/' directory")
    print("You can now test these with the neural automata using demo_neural_automata.py")


if __name__ == "__main__":
    main()
