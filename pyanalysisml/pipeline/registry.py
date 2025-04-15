"""
Registry mechanism for pipeline components.

This module provides functionality for registering and retrieving
pipeline components for use in the ML pipeline.
"""

from typing import Any, Callable, Dict, List, Optional, Type


class ComponentRegistry:
    """
    Registry for pipeline components.
    
    This class provides a mechanism to register and retrieve pipeline 
    components by name. It allows for dynamic instantiation of components
    based on configuration.
    
    Attributes:
        _components: Dictionary mapping component names to their classes
    """
    
    def __init__(self):
        """Initialize an empty component registry."""
        self._components: Dict[str, Type] = {}
    
    def register(self, component: Optional[Type] = None, *, name: Optional[str] = None) -> Callable:
        """
        Register a component with the registry.
        
        Can be used as a decorator:
            @component_registry.register(name="my_component")
            class MyComponent:
                pass
                
        Or called directly:
            component_registry.register(MyComponent, name="my_component")
        
        Args:
            component: The component class to register
            name: Optional name for the component. If not provided, the class name is used.
            
        Returns:
            The decorator function if used as decorator, otherwise the component class
        """
        def decorator(cls):
            component_name = name or cls.__name__
            self._components[component_name] = cls
            return cls
        
        if component is None:
            # Used as decorator with optional name parameter
            return decorator
        else:
            # Used as function call
            component_name = name or component.__name__
            self._components[component_name] = component
            return component
    
    def get(self, name: str) -> Optional[Type]:
        """
        Get a component by name.
        
        Args:
            name: The name of the component to retrieve
            
        Returns:
            The component class if found, None otherwise
        """
        return self._components.get(name)
    
    def list_components(self) -> Dict[str, Type]:
        """
        List all registered components.
        
        Returns:
            Dictionary of registered components, mapping names to classes
        """
        return self._components.copy()


# Singleton instance of the component registry
component_registry = ComponentRegistry() 