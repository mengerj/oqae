"""
Calculator module demonstrating clean architecture and TDD principles.

This module provides basic arithmetic operations with proper error handling,
type hints, and separation of concerns.
"""

from typing import Union

Number = Union[int, float]


class CalculatorError(Exception):
    """Base exception for calculator operations."""

    pass


class DivisionByZeroError(CalculatorError):
    """Raised when attempting to divide by zero."""

    pass


class Calculator:
    """
    A calculator class providing basic arithmetic operations.

    This class demonstrates:
    - Clean architecture with single responsibility
    - Proper error handling
    - Type hints for better AI assistance
    - Comprehensive docstrings
    """

    def add(self, a: Number, b: Number) -> Number:
        """
        Add two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Sum of the two numbers

        Example:
            >>> calc = Calculator()
            >>> calc.add(2, 3)
            5
        """
        return a + b

    def subtract(self, a: Number, b: Number) -> Number:
        """
        Subtract second number from first.

        Args:
            a: First number (minuend)
            b: Second number (subtrahend)

        Returns:
            Difference of the two numbers

        Example:
            >>> calc = Calculator()
            >>> calc.subtract(5, 3)
            2
        """
        return a - b

    def multiply(self, a: Number, b: Number) -> Number:
        """
        Multiply two numbers.

        Args:
            a: First number
            b: Second number

        Returns:
            Product of the two numbers

        Example:
            >>> calc = Calculator()
            >>> calc.multiply(4, 3)
            12
        """
        return a * b

    def divide(self, a: Number, b: Number) -> float:
        """
        Divide first number by second.

        Args:
            a: Dividend
            b: Divisor

        Returns:
            Quotient of the division

        Raises:
            DivisionByZeroError: If divisor is zero

        Example:
            >>> calc = Calculator()
            >>> calc.divide(10, 2)
            5.0
        """
        if b == 0:
            raise DivisionByZeroError("Cannot divide by zero")
        return a / b

    def power(self, base: Number, exponent: Number) -> Number:
        """
        Raise base to the power of exponent.

        Args:
            base: Base number
            exponent: Exponent

        Returns:
            Result of base raised to exponent

        Example:
            >>> calc = Calculator()
            >>> calc.power(2, 3)
            8
        """
        return base**exponent
