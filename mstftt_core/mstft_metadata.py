# mstftt_core/mstft_metadata.py

import datetime

class MSTFTMetadata:
    def __init__(self):
        # Base metadata for the meta scaffold.
        self.metadata = {
            "metadata": {
                "name": "MetaScaffold Metadata",
                "description": "Comprehensive metadata integrating meta-scaffolding principles with dynamic taxonomy transformation.",
                "value": {
                    "version": "2.0.0",
                    "lastUpdated": datetime.datetime.now().strftime("%d.%m.%Y"),
                    "purpose": ("To serve as a dynamic and adaptable taxonomy system for LLM agents, "
                                "enabling robust meta-prompting and iterative refinement."),
                    "usage": ("For advanced knowledge organization, hierarchical classification, LLM adaptation, "
                              "and AI-driven decision-making.")
                },
                "corePrinciples": [
                    {
                        "name": "Domain Independence",
                        "description": "Ensures taxonomies remain unbiased and applicable across diverse domains."
                    },
                    {
                        "name": "Structural Consistency",
                        "description": "Guarantees a logical and coherent organization of taxonomic elements."
                    },
                    {
                        "name": "Flexible Adaptation",
                        "description": "Enables dynamic restructuring in response to evolving knowledge and contexts."
                    }
                ]
            },
            "data": {
                "name": "Taxonomy Data",
                "description": ("Defines the hierarchical structure and classification criteria used for taxonomy "
                                "construction. This section can be dynamically populated with domain-specific data."),
                "subsections": []  # This can be populated later with actual taxonomy data.
            },
            "metaPrompts": {
                "name": "Base Meta Instructions",
                "description": "Static meta instructions for taxonomy validation and transformation.",
                "subsections": []  # Base prompts (if any) can be added here.
            }
        }
        # Additional validation rules, classification criteria, and adaptation methods.
        self.validation_rules = {
            "cyclic_dependency_check": "Ensure no cyclic relationships exist within the taxonomy.",
            "hierarchical_integrity": "Validate that each taxonomy level is clearly defined and non-overlapping."
        }
        self.classification_criteria = {
            "structural_integrity": "Taxonomic elements must adhere to defined structural rules.",
            "semantic_coherence": "Concepts should be semantically consistent and well-aligned."
        }
        self.adaptation_methods = {
            "recursive_meta_prompting": "Utilize iterative meta-prompts to refine taxonomy outputs.",
            "dynamic_model_selection": "Select optimal models based on task complexity and domain requirements."
        }
        # Generate dynamic meta-prompts based on core principles.
        self.meta_prompts = self._generate_meta_prompts()

    def _generate_meta_prompts(self):
        """
        Dynamically generate meta-prompts by combining the core principles
        with default instructions. This function simulates the 'magic' of meta-prompting.
        """
        core_principles = self.metadata["metadata"]["corePrinciples"]
        prompts = {}
        # Prompt for grouping properties.
        prompts["property_grouping"] = (
            "Identify property groups using the following core principles: " +
            ", ".join([p["name"] for p in core_principles]) + "."
        )
        # Prompt for extracting key aspects.
        prompts["key_aspects"] = (
            "Extract key aspects that reflect these principles: " +
            ", ".join([p["description"] for p in core_principles]) + "."
        )
        # Prompt for discovering rare features.
        prompts["rare_info"] = "Discover rare taxonomical features that challenge conventional classification."
        
        meta_prompts_section = {
            "name": "Dynamic Meta Prompts",
            "description": "Auto-generated meta prompts based on core principles and scientific insights.",
            "subsections": []
        }
        for key, prompt in prompts.items():
            meta_prompts_section["subsections"].append({
                "name": key,
                "description": prompt,
                "value": prompt
            })
        return meta_prompts_section

    def get_metadata(self):
        """Return the base metadata information."""
        return self.metadata.get("metadata", {})

    def get_taxonomy_data(self):
        """Return the taxonomy data configuration."""
        return self.metadata.get("data", {})

    def get_meta_prompts(self):
        """
        Return a merged dictionary of base meta-prompts and the dynamically generated ones.
        """
        base_prompts = self.metadata.get("metaPrompts", {})
        # Merge base prompts with dynamic prompts.
        merged = base_prompts.copy()
        merged.update(self.meta_prompts)
        return merged

    def get_validation_rules(self):
        """Return the validation rules for the taxonomy."""
        return self.validation_rules

    def get_classification_criteria(self):
        """Return the classification criteria used in taxonomy construction."""
        return self.classification_criteria

    def get_adaptation_methods(self):
        """Return the methods for adapting and refining the taxonomy."""
        return self.adaptation_methods

    def get_section(self, section_name):
        """
        Retrieve a specific section by name.
        Sections include: validationRules, classificationCriteria, adaptationMethods, metaPrompts.
        """
        if section_name == "validationRules":
            return self.validation_rules
        elif section_name == "classificationCriteria":
            return self.classification_criteria
        elif section_name == "adaptationMethods":
            return self.adaptation_methods
        elif section_name == "metaPrompts":
            return self.get_meta_prompts()
        else:
            return {}
