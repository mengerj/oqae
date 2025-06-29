"""
Test module for calculator functionality.

This module demonstrates:
- Test-driven development (TDD) principles
- Comprehensive test coverage
- Clear test organization
- Proper use of pytest fixtures and parametrization
"""

import pytest

from src.calculator import Calculator, CalculatorError, DivisionByZeroError


class TestCalculator:
    """Test suite for Calculator class."""

    @pytest.fixture
    def calculator(self) -> Calculator:
        """Fixture providing a Calculator instance."""
        return Calculator()

    def test_calculator_instantiation(self, calculator: Calculator) -> None:
        """Test that Calculator can be instantiated."""
        assert isinstance(calculator, Calculator)

    # Addition tests
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (2, 3, 5),
            (-1, 1, 0),
            (0, 0, 0),
            (10.5, 2.5, 13.0),
            (-5, -3, -8),
        ],
    )
    def test_add(
        self, calculator: Calculator, a: float, b: float, expected: float
    ) -> None:
        """Test addition with various number combinations."""
        result = calculator.add(a, b)
        assert result == expected

    # Subtraction tests
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (5, 3, 2),
            (0, 5, -5),
            (-2, -3, 1),
            (10.5, 5.5, 5.0),
            (1, 1, 0),
        ],
    )
    def test_subtract(
        self, calculator: Calculator, a: float, b: float, expected: float
    ) -> None:
        """Test subtraction with various number combinations."""
        result = calculator.subtract(a, b)
        assert result == expected

    # Multiplication tests
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (3, 4, 12),
            (0, 5, 0),
            (-2, 3, -6),
            (2.5, 4, 10.0),
            (-1, -1, 1),
        ],
    )
    def test_multiply(
        self, calculator: Calculator, a: float, b: float, expected: float
    ) -> None:
        """Test multiplication with various number combinations."""
        result = calculator.multiply(a, b)
        assert result == expected

    # Division tests
    @pytest.mark.parametrize(
        "a, b, expected",
        [
            (10, 2, 5.0),
            (0, 5, 0.0),
            (-6, 2, -3.0),
            (7, 2, 3.5),
            (-8, -4, 2.0),
        ],
    )
    def test_divide(
        self, calculator: Calculator, a: float, b: float, expected: float
    ) -> None:
        """Test division with various number combinations."""
        result = calculator.divide(a, b)
        assert result == expected

    def test_divide_by_zero_raises_exception(self, calculator: Calculator) -> None:
        """Test that division by zero raises DivisionByZeroError."""
        with pytest.raises(DivisionByZeroError, match="Cannot divide by zero"):
            calculator.divide(5, 0)

    def test_division_by_zero_error_is_calculator_error(self) -> None:
        """Test that DivisionByZeroError is a subclass of CalculatorError."""
        assert issubclass(DivisionByZeroError, CalculatorError)

    # Power tests
    @pytest.mark.parametrize(
        "base, exponent, expected",
        [
            (2, 3, 8),
            (5, 0, 1),
            (4, 0.5, 2.0),
            (-2, 2, 4),
            (10, -1, 0.1),
        ],
    )
    def test_power(
        self, calculator: Calculator, base: float, exponent: float, expected: float
    ) -> None:
        """Test power operation with various combinations."""
        result = calculator.power(base, exponent)
        assert result == pytest.approx(expected)

    def test_complex_calculation_sequence(self, calculator: Calculator) -> None:
        """Test a sequence of operations to ensure state independence."""
        # Each operation should not affect others
        assert calculator.add(2, 3) == 5
        assert calculator.multiply(4, 5) == 20
        assert calculator.subtract(10, 3) == 7
        assert calculator.divide(8, 2) == 4.0
        assert calculator.power(2, 3) == 8

    def test_type_hints_work_with_integers_and_floats(
        self, calculator: Calculator
    ) -> None:
        """Test that type hints work correctly with both int and float."""
        # Test with integers
        int_result = calculator.add(1, 2)
        assert isinstance(int_result, int)
        assert int_result == 3

        # Test with floats
        float_result = calculator.add(1.5, 2.5)
        assert isinstance(float_result, float)
        assert float_result == 4.0

        # Test mixed types
        mixed_result = calculator.add(1, 2.5)
        assert isinstance(mixed_result, float)
        assert mixed_result == 3.5
