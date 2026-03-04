#!/usr/bin/env python3
"""
Developer Setup Verification Script

This script verifies that the development environment is properly configured
and all APIs are working correctly. It helps new developers get started
quickly and ensures existing developers haven't broken anything.

Usage:
    python scripts/verify_setup.py
"""

import sys
import importlib
import traceback
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    required_version = (3, 10)
    
    if version >= required_version:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (>= {required_version[0]}.{required_version[1]})")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (< {required_version[0]}.{required_version[1]})")
        return False


def check_core_dependencies():
    """Check that core dependencies are available."""
    print("\n📦 Checking core dependencies...")
    
    core_deps = [
        'numpy',
        'pandas', 
        'scipy',
        'xarray',
        'zarr',
        'dask',
        'fsspec',
        'plotly'
    ]
    
    all_good = True
    for dep in core_deps:
        try:
            module = importlib.import_module(dep)
            version = getattr(module, '__version__', 'unknown')
            print(f"✅ {dep} ({version})")
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            all_good = False
    
    return all_good


def check_api_imports():
    """Check that all API classes can be imported."""
    print("\n🔬 Checking API imports...")
    
    apis = [
        'Ag3',
        'Af1', 
        'Amin1',
        'Adir1',
        'Adar1',
        'Pf7',
        'Pf8',
        'Pv4'
    ]
    
    all_good = True
    for api in apis:
        try:
            # Try to import the main module
            import malariagen_data
            api_class = getattr(malariagen_data, api)
            print(f"✅ {api}")
        except (ImportError, AttributeError) as e:
            print(f"❌ {api}: {e}")
            all_good = False
    
    return all_good


def check_documentation_files():
    """Check that all documentation files exist."""
    print("\n📚 Checking documentation files...")
    
    doc_files = [
        'docs/source/index.rst',
        'docs/source/Ag3.rst',
        'docs/source/Af1.rst', 
        'docs/source/Amin1.rst',
        'docs/source/Adir1.rst',
        'docs/source/Adar1.rst',
        'docs/source/Pf7.rst',
        'docs/source/Pf8.rst',
        'docs/source/Pv4.rst'
    ]
    
    all_good = True
    for doc_file in doc_files:
        if Path(doc_file).exists():
            print(f"✅ {doc_file}")
        else:
            print(f"❌ {doc_file} (missing)")
            all_good = False
    
    return all_good


def check_basic_functionality():
    """Check basic functionality of APIs."""
    print("\n🧪 Checking basic functionality...")
    
    try:
        import malariagen_data
        
        # Test that we can create API instances (without actually connecting to data)
        print("Testing API instantiation...")
        
        # Test a simple API that should work locally
        try:
            pf7 = malariagen_data.Pf7(url="dummy://test")  # Should not fail immediately
            print("✅ Pf7 instantiation works")
        except Exception as e:
            # Expected to fail with dummy URL, but should fail during connection, not import
            if "dummy" in str(e).lower() or "url" in str(e).lower():
                print("✅ Pf7 instantiation works (failed at connection as expected)")
            else:
                print(f"❌ Pf7 instantiation failed unexpectedly: {e}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False


def check_git_hooks():
    """Check if git hooks are properly configured."""
    print("\n🪝 Checking git hooks...")
    
    try:
        import subprocess
        result = subprocess.run(['git', 'config', '--get', 'core.hooksPath'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            hooks_path = result.stdout.strip()
            print(f"✅ Git hooks path: {hooks_path}")
        else:
            print("⚠️  No custom git hooks path configured")
        
        # Check if pre-commit is installed
        try:
            result = subprocess.run(['pre-commit', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ Pre-commit: {result.stdout.strip()}")
            else:
                print("⚠️  Pre-commit not available")
        except FileNotFoundError:
            print("⚠️  Pre-commit not installed")
            
        return True
        
    except Exception as e:
        print(f"❌ Git hooks check failed: {e}")
        return False


def main():
    """Run all verification checks."""
    print("🚀 MalariaGEN Data Python - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Core Dependencies", check_core_dependencies),
        ("API Imports", check_api_imports),
        ("Documentation Files", check_documentation_files),
        ("Basic Functionality", check_basic_functionality),
        ("Git Hooks", check_git_hooks),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:<8} {name}")
    
    print(f"\nResults: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Your development environment is ready.")
        return 0
    else:
        print("⚠️  Some checks failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
