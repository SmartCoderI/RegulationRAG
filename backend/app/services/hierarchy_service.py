"""
Regulatory Hierarchy Service

Handles multi-jurisdictional regulation conflicts and precedence.
Implements the principle: District > City > County > State
"""

from typing import List, Dict, Optional, Tuple, Any
import re
from dataclasses import dataclass
from app.models.schemas import SourceDocument

@dataclass
class JurisdictionHierarchy:
    """Defines the regulatory hierarchy levels"""
    DISTRICT = 1
    CITY = 2  
    COUNTY = 3
    STATE = 4
    
    @classmethod
    def get_priority(cls, level: str) -> int:
        """Get numeric priority for jurisdiction level"""
        mapping = {
            "district": cls.DISTRICT,
            "special_district": cls.DISTRICT,
            "city": cls.CITY,
            "municipal": cls.CITY,
            "county": cls.COUNTY,
            "state": cls.STATE,
            "california": cls.STATE
        }
        return mapping.get(level.lower(), cls.STATE)
    
    @classmethod
    def get_level_name(cls, priority: int) -> str:
        """Get human-readable name for priority level"""
        mapping = {
            cls.DISTRICT: "District/Special District",
            cls.CITY: "City/Municipal", 
            cls.COUNTY: "County",
            cls.STATE: "State"
        }
        return mapping.get(priority, "Unknown")

class RegulationConflictDetector:
    """Detects conflicts between regulations at different jurisdiction levels"""
    
    def __init__(self):
        # Patterns to extract numeric requirements
        self.numeric_patterns = {
            'setback': r'(?:setback|separation).*?(\d+(?:\.\d+)?)\s*(?:feet|ft|meters?|m)',
            'height': r'(?:height|tall).*?(\d+(?:\.\d+)?)\s*(?:feet|ft|meters?|m|stories?)',
            'parking': r'(?:parking).*?(\d+(?:\.\d+)?)\s*(?:spaces?|stalls?)',
            'lot_size': r'(?:lot size|area).*?(\d+(?:\.\d+)?)\s*(?:sq\.?\s*ft|square feet|acres?)',
            'coverage': r'(?:coverage|coverage ratio).*?(\d+(?:\.\d+)?)%?',
            'density': r'(?:density).*?(\d+(?:\.\d+)?)\s*(?:units?|dwelling)',
        }
    
    def extract_requirements(self, content: str) -> Dict[str, float]:
        """Extract numeric requirements from regulation text"""
        requirements = {}
        
        for req_type, pattern in self.numeric_patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                try:
                    value = float(match.group(1))
                    requirements[f"{req_type}_value"] = value
                except (ValueError, IndexError):
                    continue
                    
        return requirements
    
    def detect_conflicts(self, documents: List[SourceDocument]) -> List[SourceDocument]:
        """Detect conflicts between regulations and mark controlling authority"""
        
        # Group documents by regulation type and jurisdiction
        by_type = {}
        for doc in documents:
            reg_type = doc.regulation_type or "general"
            if reg_type not in by_type:
                by_type[reg_type] = []
            by_type[reg_type].append(doc)
        
        # Check for conflicts within each regulation type
        for reg_type, docs in by_type.items():
            if len(docs) <= 1:
                continue
                
            # Extract requirements for each document
            for doc in docs:
                doc.numeric_requirements = self.extract_requirements(doc.content)
                doc.jurisdiction_priority = JurisdictionHierarchy.get_priority(
                    doc.jurisdiction_level or "state"
                )
            
            # Find conflicts
            conflicts = self._find_requirement_conflicts(docs)
            
            # Mark controlling authority (lowest priority number = highest authority)
            if conflicts:
                controlling_doc = min(docs, key=lambda d: d.jurisdiction_priority or 999)
                controlling_doc.controlling_authority = True
                
                # Mark conflicting documents
                for doc in docs:
                    if doc != controlling_doc:
                        doc.conflicts_with = [controlling_doc.jurisdiction_name or "Higher Authority"]
        
        return documents
    
    def _find_requirement_conflicts(self, documents: List[SourceDocument]) -> bool:
        """Check if documents have conflicting numeric requirements"""
        requirement_types = set()
        for doc in documents:
            if doc.numeric_requirements:
                requirement_types.update(doc.numeric_requirements.keys())
        
        for req_type in requirement_types:
            values = []
            for doc in documents:
                if doc.numeric_requirements and req_type in doc.numeric_requirements:
                    values.append(doc.numeric_requirements[req_type])
            
            # If we have different values for same requirement type, it's a conflict
            if len(set(values)) > 1:
                return True
                
        return False

class HierarchicalRegulatoryService:
    """Main service for handling hierarchical regulations"""
    
    def __init__(self):
        self.conflict_detector = RegulationConflictDetector()
        self.hierarchy = JurisdictionHierarchy()
    
    def organize_by_hierarchy(self, documents: List[SourceDocument], 
                            question: str) -> Dict[str, List[SourceDocument]]:
        """Organize documents by jurisdiction hierarchy"""
        
        # Detect conflicts and mark controlling authorities
        documents = self.conflict_detector.detect_conflicts(documents)
        
        # Group by jurisdiction level
        hierarchy_groups = {
            "district": [],
            "city": [], 
            "county": [],
            "state": []
        }
        
        for doc in documents:
            level = doc.jurisdiction_level or "state"
            if level in hierarchy_groups:
                hierarchy_groups[level].append(doc)
            else:
                hierarchy_groups["state"].append(doc)  # Default to state level
        
        # Sort within each group by relevance score
        for level in hierarchy_groups:
            hierarchy_groups[level].sort(key=lambda d: d.score or 0, reverse=True)
        
        return hierarchy_groups
    
    def generate_hierarchical_response(self, hierarchy_groups: Dict[str, List[SourceDocument]], 
                                     question: str) -> Dict[str, Any]:
        """Generate response with hierarchy explanation"""
        
        controlling_regulations = []
        supporting_regulations = []
        conflicts_found = False
        
        # Find controlling regulations (highest priority level with content)
        for level in ["district", "city", "county", "state"]:
            if hierarchy_groups[level]:
                for doc in hierarchy_groups[level]:
                    if doc.controlling_authority:
                        controlling_regulations.append(doc)
                        conflicts_found = True
                    else:
                        supporting_regulations.append(doc)
                
                # If we found regulations at this level, stop looking at lower priority levels
                # unless we need to show conflicts
                if not conflicts_found:
                    controlling_regulations.extend(hierarchy_groups[level])
                break
        
        # Add remaining documents as supporting
        for level in ["district", "city", "county", "state"]:
            for doc in hierarchy_groups[level]:
                if doc not in controlling_regulations:
                    supporting_regulations.append(doc)
        
        return {
            "controlling_regulations": controlling_regulations,
            "supporting_regulations": supporting_regulations,
            "conflicts_detected": conflicts_found,
            "hierarchy_explanation": self._generate_hierarchy_explanation(
                hierarchy_groups, conflicts_found
            )
        }
    
    def _generate_hierarchy_explanation(self, hierarchy_groups: Dict[str, List[SourceDocument]], 
                                      conflicts_found: bool) -> str:
        """Generate explanation of regulatory hierarchy"""
        
        active_levels = [level for level, docs in hierarchy_groups.items() if docs]
        
        if not active_levels:
            return "No applicable regulations found."
        
        explanation = "**Regulatory Hierarchy Applied:**\n\n"
        
        if conflicts_found:
            explanation += "⚖️ **Conflicts detected** - More specific regulations take precedence:\n\n"
        
        level_names = {
            "district": "🏛️ **Special District/Master Plan**",
            "city": "🏙️ **City of Sunnyvale**", 
            "county": "🏞️ **Santa Clara County**",
            "state": "🌟 **State of California**"
        }
        
        for level in ["district", "city", "county", "state"]:
            if level in active_levels:
                docs = hierarchy_groups[level]
                explanation += f"{level_names[level]} ({len(docs)} regulation{'s' if len(docs) != 1 else ''})\n"
                
                controlling_docs = [d for d in docs if d.controlling_authority]
                if controlling_docs:
                    explanation += f"   ✅ **Controls** - These regulations take precedence\n"
                elif conflicts_found:
                    explanation += f"   ⚠️ Superseded by higher authority\n"
                
                explanation += "\n"
        
        if conflicts_found:
            explanation += "\n💡 **Note:** When regulations conflict, the most specific (district/local) authority controls, followed by city, county, then state regulations."
        
        return explanation 