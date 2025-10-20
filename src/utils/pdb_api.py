# """
# RCSB PDB API Integration
# Fetches protein structure information from PDB database
# """

# import requests
# import logging
# from typing import Dict, Optional, List


# class PDBFetcher:
#     """
#     Fetch protein data from RCSB PDB
#     """
    
#     BASE_URL = "https://data.rcsb.org/rest/v1/core"
#     SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
#     def __init__(self):
#         self.logger = logging.getLogger(__name__)
    
#     def fetch_pdb_info(self, pdb_id: str) -> Optional[Dict]:
#         """
#         Fetch basic PDB information
        
#         Args:
#             pdb_id: PDB ID (e.g., '2SRC')
            
#         Returns:
#             Dictionary with PDB information
#         """
#         pdb_id = pdb_id.upper().strip()
        
#         try:
#             url = f"{self.BASE_URL}/entry/{pdb_id}"
#             response = requests.get(url, timeout=10)
            
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # Extract relevant information
#                 info = {
#                     'pdb_id': pdb_id,
#                     'title': data.get('struct', {}).get('title', 'N/A'),
#                     'experimental_method': data.get('exptl', [{}])[0].get('method', 'N/A'),
#                     'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0],
#                     'release_date': data.get('rcsb_accession_info', {}).get('initial_release_date', 'N/A'),
#                     'organism': self._extract_organism(data),
#                     'num_chains': len(data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])),
#                     'status': 'Found'
#                 }
                
#                 self.logger.info(f"✓ Found PDB: {pdb_id}")
#                 return info
            
#             elif response.status_code == 404:
#                 self.logger.warning(f"PDB {pdb_id} not found")
#                 return {'pdb_id': pdb_id, 'status': 'Not Found'}
            
#             else:
#                 self.logger.error(f"API error: {response.status_code}")
#                 return {'pdb_id': pdb_id, 'status': 'Error'}
        
#         except Exception as e:
#             self.logger.error(f"Failed to fetch PDB {pdb_id}: {e}")
#             return {'pdb_id': pdb_id, 'status': 'Error', 'error': str(e)}
    
#     def _extract_organism(self, data: Dict) -> str:
#         """Extract organism name from PDB data"""
#         try:
#             entities = data.get('rcsb_entity_source_organism', [])
#             if entities:
#                 return entities[0].get('ncbi_scientific_name', 'N/A')
#             return 'N/A'
#         except:
#             return 'N/A'
    
#     def get_binding_sites(self, pdb_id: str) -> List[Dict]:
#         """
#         Get binding site information
        
#         Args:
#             pdb_id: PDB ID
            
#         Returns:
#             List of binding sites
#         """
#         pdb_id = pdb_id.upper().strip()
        
#         try:
#             url = f"{self.BASE_URL}/entry/{pdb_id}"
#             response = requests.get(url, timeout=10)
            
#             if response.status_code == 200:
#                 data = response.json()
                
#                 # Extract binding site info
#                 sites = []
#                 binding_data = data.get('rcsb_binding_affinity', [])
                
#                 for site in binding_data:
#                     sites.append({
#                         'ligand': site.get('comp_id', 'Unknown'),
#                         'type': site.get('type', 'N/A'),
#                         'value': site.get('value', 'N/A'),
#                         'unit': site.get('unit', 'N/A')
#                     })
                
#                 return sites if sites else [{'info': 'No binding site data available'}]
            
#             return [{'info': 'Could not fetch binding site data'}]
        
#         except Exception as e:
#             self.logger.error(f"Error fetching binding sites: {e}")
#             return [{'error': str(e)}]
    
#     def get_ligands(self, pdb_id: str) -> List[Dict]:
#         """
#         Get ligand information
        
#         Args:
#             pdb_id: PDB ID
            
#         Returns:
#             List of ligands
#         """
#         pdb_id = pdb_id.upper().strip()
        
#         try:
#             url = f"{self.BASE_URL}/entry/{pdb_id}"
#             response = requests.get(url, timeout=10)
            
#             if response.status_code == 200:
#                 data = response.json()
                
#                 ligands = []
#                 # Get non-polymer entities (ligands)
#                 nonpolymer_entities = data.get('rcsb_entry_container_identifiers', {}).get('non_polymer_entity_ids', [])
                
#                 for entity_id in nonpolymer_entities[:5]:  # Limit to 5
#                     ligands.append({
#                         'entity_id': entity_id,
#                         'type': 'Ligand'
#                     })
                
#                 return ligands if ligands else [{'info': 'No ligand data available'}]
            
#             return [{'info': 'Could not fetch ligand data'}]
        
#         except Exception as e:
#             self.logger.error(f"Error fetching ligands: {e}")
#             return [{'error': str(e)}]
    
#     def validate_pdb_id(self, pdb_id: str) -> bool:
#         """
#         Quick validation of PDB ID format and existence
        
#         Args:
#             pdb_id: PDB ID to validate
            
#         Returns:
#             True if valid and exists
#         """
#         pdb_id = pdb_id.upper().strip()
        
#         # Check format (4 characters)
#         if len(pdb_id) != 4:
#             return False
        
#         # Check if exists
#         try:
#             url = f"{self.BASE_URL}/entry/{pdb_id}"
#             response = requests.head(url, timeout=5)
#             return response.status_code == 200
#         except:
#             return False


# def main():
#     """Example usage"""
#     fetcher = PDBFetcher()
    
#     # Test with 2SRC (Src kinase)
#     print("\nFetching PDB: 2SRC (Src Kinase)")
#     print("="*50)
    
#     info = fetcher.fetch_pdb_info('2SRC')
    
#     if info and info.get('status') == 'Found':
#         print(f"Title: {info['title']}")
#         print(f"Method: {info['experimental_method']}")
#         print(f"Resolution: {info['resolution']} Å")
#         print(f"Organism: {info['organism']}")
#         print(f"Released: {info['release_date']}")
        
#         # Get binding sites
#         print("\nBinding Sites:")
#         sites = fetcher.get_binding_sites('2SRC')
#         for site in sites:
#             print(f"  - {site}")
        
#         # Get ligands
#         print("\nLigands:")
#         ligands = fetcher.get_ligands('2SRC')
#         for lig in ligands:
#             print(f"  - {lig}")
#     else:
#         print(f"Status: {info.get('status', 'Error')}")
    
#     print("\n" + "="*50)


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     main()

"""
RCSB PDB API Integration
Fetches protein structure information from PDB database
"""

import requests
import logging
from typing import Dict, Optional, List


class PDBFetcher:
    """
    Fetch protein data from RCSB PDB
    """
    
    BASE_URL = "https://data.rcsb.org/rest/v1/core"
    SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_pdb_info(self, pdb_id: str) -> Optional[Dict]:
        """
        Fetch basic PDB information
        
        Args:
            pdb_id: PDB ID (e.g., '2SRC')
            
        Returns:
            Dictionary with PDB information
        """
        pdb_id = pdb_id.upper().strip()
        
        try:
            url = f"{self.BASE_URL}/entry/{pdb_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract relevant information
                info = {
                    'pdb_id': pdb_id,
                    'title': data.get('struct', {}).get('title', 'N/A'),
                    'experimental_method': data.get('exptl', [{}])[0].get('method', 'N/A'),
                    'resolution': data.get('rcsb_entry_info', {}).get('resolution_combined', [None])[0],
                    'release_date': data.get('rcsb_accession_info', {}).get('initial_release_date', 'N/A'),
                    'organism': self._extract_organism(data),
                    'num_chains': len(data.get('rcsb_entry_container_identifiers', {}).get('polymer_entity_ids', [])),
                    'status': 'Found'
                }
                
                self.logger.info(f"✓ Found PDB: {pdb_id}")
                return info
            
            elif response.status_code == 404:
                self.logger.warning(f"PDB {pdb_id} not found")
                return {'pdb_id': pdb_id, 'status': 'Not Found'}
            
            else:
                self.logger.error(f"API error: {response.status_code}")
                return {'pdb_id': pdb_id, 'status': 'Error'}
        
        except Exception as e:
            self.logger.error(f"Failed to fetch PDB {pdb_id}: {e}")
            return {'pdb_id': pdb_id, 'status': 'Error', 'error': str(e)}
    
    def _extract_organism(self, data: Dict) -> str:
        """Extract organism name from PDB data"""
        try:
            entities = data.get('rcsb_entity_source_organism', [])
            if entities:
                return entities[0].get('ncbi_scientific_name', 'N/A')
            return 'N/A'
        except:
            return 'N/A'
    
    def get_binding_sites(self, pdb_id: str) -> List[Dict]:
        """
        Get binding site information
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            List of binding sites
        """
        pdb_id = pdb_id.upper().strip()
        
        try:
            url = f"{self.BASE_URL}/entry/{pdb_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract binding site info
                sites = []
                binding_data = data.get('rcsb_binding_affinity', [])
                
                for site in binding_data:
                    sites.append({
                        'ligand': site.get('comp_id', 'Unknown'),
                        'type': site.get('type', 'N/A'),
                        'value': site.get('value', 'N/A'),
                        'unit': site.get('unit', 'N/A')
                    })
                
                return sites if sites else [{'info': 'No binding site data available'}]
            
            return [{'info': 'Could not fetch binding site data'}]
        
        except Exception as e:
            self.logger.error(f"Error fetching binding sites: {e}")
            return [{'error': str(e)}]
    
    def get_ligands(self, pdb_id: str) -> List[Dict]:
        """
        Get ligand information
        
        Args:
            pdb_id: PDB ID
            
        Returns:
            List of ligands
        """
        pdb_id = pdb_id.upper().strip()
        
        try:
            url = f"{self.BASE_URL}/entry/{pdb_id}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                ligands = []
                # Get non-polymer entities (ligands)
                nonpolymer_entities = data.get('rcsb_entry_container_identifiers', {}).get('non_polymer_entity_ids', [])
                
                for entity_id in nonpolymer_entities[:5]:  # Limit to 5
                    ligands.append({
                        'entity_id': entity_id,
                        'type': 'Ligand'
                    })
                
                return ligands if ligands else [{'info': 'No ligand data available'}]
            
            return [{'info': 'Could not fetch ligand data'}]
        
        except Exception as e:
            self.logger.error(f"Error fetching ligands: {e}")
            return [{'error': str(e)}]
    
    def validate_pdb_id(self, pdb_id: str) -> bool:
        """
        Quick validation of PDB ID format and existence
        
        Args:
            pdb_id: PDB ID to validate
            
        Returns:
            True if valid and exists
        """
        pdb_id = pdb_id.upper().strip()
        
        # Check format (4 characters)
        if len(pdb_id) != 4:
            return False
        
        # Check if exists
        try:
            url = f"{self.BASE_URL}/entry/{pdb_id}"
            response = requests.head(url, timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    """Example usage"""
    fetcher = PDBFetcher()
    
    # Test with 2SRC (Src kinase)
    print("\nFetching PDB: 2SRC (Src Kinase)")
    print("="*50)
    
    info = fetcher.fetch_pdb_info('2SRC')
    
    if info and info.get('status') == 'Found':
        print(f"Title: {info['title']}")
        print(f"Method: {info['experimental_method']}")
        print(f"Resolution: {info['resolution']} Å")
        print(f"Organism: {info['organism']}")
        print(f"Released: {info['release_date']}")
        
        # Get binding sites
        print("\nBinding Sites:")
        sites = fetcher.get_binding_sites('2SRC')
        for site in sites:
            print(f"  - {site}")
        
        # Get ligands
        print("\nLigands:")
        ligands = fetcher.get_ligands('2SRC')
        for lig in ligands:
            print(f"  - {lig}")
    else:
        print(f"Status: {info.get('status', 'Error')}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()