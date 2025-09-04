# Contributing to OpenSG

We welcome contributions to OpenSG! This document provides guidelines for contributing to the project.

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## How to Contribute

### Reporting Issues

Before creating an issue, please:

1. Check if the issue has already been reported in the [Issues](https://github.com/wenbinyugroup/OpenSG/issues) section
2. Use the appropriate issue template (bug report or feature request)
3. Provide as much detail as possible, including:
   - Operating system and Python version
   - Steps to reproduce the issue
   - Expected vs. actual behavior
   - Relevant error messages or logs

### Suggesting Enhancements

We welcome suggestions for new features and improvements. When suggesting enhancements:

1. Use the feature request template
2. Clearly describe the proposed feature
3. Explain the use case and potential benefits
4. Consider implementation complexity and maintenance burden

### Contributing Code

#### Setting Up Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/OpenSG.git
   cd OpenSG
   ```

3. Create a conda environment for development:
   ```bash
   conda env create --file environment.yml
   conda activate opensg_env
   ```

4. Install the package in development mode (Currently this is done by default by step 3., but this may change in the future.):
   ```bash
   pip install -e .
   ```

#### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number
   ```

2. Make your changes following our coding standards (see below)

3. Add tests for new functionality or bug fixes

4. Run the test suite to ensure everything works:
   ```bash
   python -m pytest opensg/tests/
   ```

5. Update documentation if necessary

6. Commit your changes with clear, descriptive commit messages

7. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

8. Create a Pull Request using the provided template

#### Coding Standards

- Follow PEP 8 style guidelines for Python code
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Keep functions focused and reasonably sized
- Add comments for complex algorithms or non-obvious code

#### Testing Requirements

- All new features must include appropriate tests
- Bug fixes must include tests that verify the fix
- Maintain or improve test coverage
- Tests should be deterministic and not depend on external resources
- Use descriptive test names that explain what is being tested

#### Documentation Requirements

- Update docstrings for any modified functions
- Update README.md if installation or usage instructions change
- Update API documentation if new functions are added
- Include examples for new features

### Pull Request Process

1. Ensure your PR addresses a single issue or feature
2. Use the provided pull request template
3. Include a clear description of changes
4. Reference any related issues
5. Ensure all tests pass
6. Request review from maintainers

### Review Process

- All pull requests require review from at least one maintainer
- Address review feedback promptly
- Keep discussions focused and constructive
- Be open to suggestions and alternative approaches

## Development Guidelines

### Project Structure

- `opensg/core/` - Core computation modules
- `opensg/mesh/` - Mesh handling and management
- `opensg/io/` - Input/output operations
- `opensg/utils/` - Utility functions
- `opensg/tests/` - Test suite
- `examples/` - Example scripts and tutorials
- `docs/` - Documentation source files

### Dependencies

When adding new dependencies:

1. Add to `pyproject.toml` in the appropriate section
2. Update `environment.yml`
3. Consider the impact on installation complexity
4. Ensure compatibility with existing dependencies

### Version Control

- Use meaningful commit messages
- Keep commits focused and atomic
- Use conventional commit format when possible:
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `test:` for test additions or modifications
  - `refactor:` for code refactoring

## Getting Help

If you need help with contributing:

1. Check the [documentation](https://wenbinyugroup.github.io/OpenSG/)
2. Search existing [issues](https://github.com/wenbinyugroup/OpenSG/issues)
3. Start a [discussion](https://github.com/wenbinyugroup/OpenSG/discussions)
4. Contact maintainers directly if needed

## License

By contributing to OpenSG, you agree that your contributions will be licensed under the [MIT License](LICENSE).
js
