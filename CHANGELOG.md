# Changelog

All notable changes to LEAPP (Lightweight Export Annotations for Policy Pipelines) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-10-31

### Added

#### Core Features
- **Automatic Node Merging**: Added functionality to automatically merge fully sequential nodes for optimized graph execution
- **Feedback Detection**: Implemented feedback loop detection in computational graphs with visualization support
- **Explicit Return Values**: Added ability to declare return values in `@annotate.method()` that don't exist in the original method signature. LEAPP internally modifies or creates return statements to return the specified variables
- **Mirror LEAPP Tags**: Added `annotate.mirror_leapp_tags()` function to maintain proper tracing when tensor data is duplicated without using `clone()` or `detach()`. Verifies data equivalence and transfers tags from source to target. Logs an error if data doesn't match
- **SHA256 Model Verification**: Added SHA256 hashing for model verification and integrity checking
- **Node Sequencing**: Now saves node sequencing information for better graph understanding
- **YAML Metadata Field**: Added metadata field in YAML export files for storing export environment information

#### Developer Experience
- **Better Logging**: Improved logging system with better formatting and information
- **System Information**: Added system info collection inside YAML export files

#### Visualization
- **Feedback Visualization**: Added visualization capabilities for feedback loops in graphs
- **Feedback Example Images**: Added example images showing feedback loop detection

### Changed

#### Architecture
- **Reorganized Graph Components**: 
  - Moved graph GUI functionality to `leapp/leapp_graph/` module
  - Created new `leapp_graph` package structure with:
    - `graph_element.py` - Core graph element definitions
    - `leapp_graph.py` - Main graph operations
    - `leapp_combination_node.py` - Node combination logic
    - `node_context.py` - Node context management (moved from `leapp/node_context.py`)
- **Export Manager Refactoring**: Significant refactoring of export_manager.py for improved clarity and robustness
- **Backend Improvements**: Enhanced torch.py backend with better structure (~228 lines changed)
- **Utility Enhancements**: Expanded utils.py with additional helper functions (~453 lines changed)

#### Dependencies
- **Removed TensorDict Dependency**: Unified dictionary detection methods to eliminate tensordict dependency
- **Python Version**: Downgraded minimum Python version to 3.8 for better stability with external libraries

#### Graph Processing
- **Node Merging Logic**: Disabled fusing for node merging; changed auto-merging logic for better clarity and robustness
- **Environment Constants**: declared environment constants now buffer the value before node tracing for more utility and more predictable behavior

#### Code Cleanup
- **Removed Untagged IO Method**: Cleaned up unused untagged IO method functionality

### Fixed

#### Bug Fixes
- **Isaac Lab Compatibility**: Fixed bugs for Isaac Lab integration
- **nn.Module header geneartion**: Fixed bug in nested tensor function header generation
- **Kwargs Support**: Fixed support for keyword argument inputs in traced functions
- **Multiple Unnamed Returns**: Fixed support for functions with multiple unnamed return values
- **Typos**: Fixed various typos throughout the codebase

### Development

#### Infrastructure
- **CI/CD Updates**: Updated `.gitlab-ci.yml` configuration
- **Build System**: 
  - Updated pyproject.toml configuration
  - Added proper package structure with setuptools
- **Requirements**: Added requirements.txt for examples