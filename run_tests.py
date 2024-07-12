import unittest

if __name__ == '__main__':
    # Discover and run all tests
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests')
    
    runner = unittest.TextTestRunner()
    runner.run(test_suite)