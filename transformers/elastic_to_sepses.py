import re
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from models.event import SEPSesLogEvent
from models.resource import (
    SEPSesHost, SEPSesProcess, SEPSesUser, 
    SEPSesFile, SEPSesNetworkFlow
)
from models.threat import MITRETTP, AttackChain
from config.settings import settings

class ElasticToSEPSesTransformer:
    """
    Transforms raw Elasticsearch alerts into SEPSes ontology-compliant objects
    Uses ONLY real SEPSes classes (LogEvent, Host, Process, ...) - NO fake "Alert" class
    """
    
    def __init__(self, base_uri: str = None):
        self.base_uri = base_uri or settings.SEPSES_BASE_URI.rstrip('#')
    
        
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse Elasticsearch timestamp string to datetime"""
        try:
            return datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        except ValueError:
            return datetime.now(timezone.utcnow())
    
    # def _extract_value(self, source: Dict, path: str, default=None):
    #     """Safely extract nested value from Elasticsearch source"""
    #     keys = path.split('.')
    #     value = source
    #     for key in keys:
    #         if isinstance(value, dict):
    #             value = value.get(key, default)
    #         else:
    #             return default
    #     return value

    def _extract_value(self, source: dict, path: str, default=None):
        """Safely extract nested value from Elasticsearch source, including dotted keys"""

        # Case 1: exact key match (ES dotted fields like 'kibana.alert.rule.threat')
        if path in source:
            return source[path]

        # Case 2: nested key resolution
        keys = path.split('.')
        value = source
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key, default)
            else:
                return default
        return value

    
    def _extract_ttps(self, threat_data: List[Dict]) -> List[MITRETTP]:
        """Extract MITRE ATT&CK techniques from kibana.alert.rule.threat."""
        ttps = []
        if not threat_data:
            return ttps
        
        for item in threat_data:
            framework = item.get('framework', '')
            if 'MITRE ATT&CK' not in framework:
                continue
            
            current_item_tactics = {}

            # Handling tactics: This section assumes 'tactic' is a list of dicts,
            # or a single dict, or potentially a string that needs special handling.
            # The original code had a potential error here if 'tactic' was not a dict/list of dicts.
            # We will process it defensively.
            tactic_input = item.get('tactic', []) # Default to empty list

            if isinstance(tactic_input, list):
                for tactic_data in tactic_input:
                    if isinstance(tactic_data, dict): # Expecting dict
                        tactic_id = tactic_data.get('id', 'unknown')
                        tactic_name = tactic_data.get('name', 'unknown')
                        current_item_tactics[tactic_id] = tactic_name
                    elif isinstance(tactic_input, str): # If the list contained a string
                        pseudo_tactic_id = tactic_input.lower().replace(' ', '_')
                        current_item_tactics[pseudo_tactic_id] = tactic_input
            elif isinstance(tactic_input, dict): # Handling case where 'tactic' is a single dict
                tactic_id = tactic_input.get('id', 'unknown')
                tactic_name = tactic_input.get('name', 'unknown')
                current_item_tactics[tactic_id] = tactic_name
            elif isinstance(tactic_input, str): # Handling case where 'tactic' is a single string
                pseudo_tactic_id = tactic_input.lower().replace(' ', '_')
                current_item_tactics[pseudo_tactic_id] = tactic_input


            # Handling techniques
            # Use .get('technique', []) to safely handle missing 'technique' key
            for technique_data in item.get('technique', []):
                # Ensure technique_data is a dictionary before proceeding
                if not isinstance(technique_data, dict):
                    continue # Skip if not a dictionary

                technique_id = technique_data.get('id', 'unknown')
                technique_name = technique_data.get('name', 'unknown')
                
                subtechnique_name = None
                # Safely get subtechnique, default to empty list
                subtechniques_list = technique_data.get('subtechnique', []) 
                
                # Check if subtechniques_list is a list and not empty
                if subtechniques_list and isinstance(subtechniques_list, list):
                    # Check if the first element is a dictionary
                    if subtechniques_list[0] and isinstance(subtechniques_list[0], dict): 
                        first_sub = subtechniques_list[0] 
                        subtechnique_name = first_sub.get('name', None)

                extracted_tactic_id = 'unknown'
                extracted_tactic_name = 'unknown'

                # Try to extract tactic ID from technique ID if it contains a dot (e.g., T1059.001)
                if '.' in technique_id:
                    potential_tactic_id = technique_id.split('.')[0]
                    if potential_tactic_id in current_item_tactics:
                        extracted_tactic_id = potential_tactic_id
                        extracted_tactic_name = current_item_tactics[potential_tactic_id]
                
                # If tactic ID is still unknown and there's exactly one tactic for this item, use it
                if extracted_tactic_id == 'unknown' and len(current_item_tactics) == 1:
                    # Get the single key-value pair from the tactics dictionary
                    tactic_id, tactic_name = list(current_item_tactics.items())[0]
                    extracted_tactic_id = tactic_id
                    extracted_tactic_name = tactic_name
                
                # Only append if we have a valid technique ID
                if technique_id != 'unknown':
                    ttp = MITRETTP(
                        technique_id=technique_id,
                        technique_name=technique_name,
                        tactic_id=extracted_tactic_id,
                        tactic_name=extracted_tactic_name,
                        subtechnique=subtechnique_name
                    )
                    ttps.append(ttp)
        return ttps
    
    def transform_log_event(self, elastic_doc: Dict[str, Any]) -> SEPSesLogEvent:
        """Transform single Elasticsearch alert into SEPSesLogEvent (NOT Alert!)"""
        source = elastic_doc.get("_source", {})
        doc_id = elastic_doc.get("_id", "unknown")
        
        # Parse timestamp
        ts_str = source.get("@timestamp", datetime.now(timezone.utc).isoformat())
        timestamp = self._parse_timestamp(ts_str)
        
        # Build SEPSes URI for this LogEvent
        event_uri = f"{self.base_uri}LogEvent/{doc_id}"
        
        # Extract MITRE TTPs
        threat_data = self._extract_value(source, "kibana.alert.rule.threat", [])
        ttps = self._extract_ttps(threat_data if isinstance(threat_data, list) else [threat_data])
        
        # Build Host entity
        host_data = source.get("host", {})
        host = None
        if host_data:
            host_name = host_data.get("name") if isinstance(host_data, dict) else str(host_data)
            host_ip = host_data.get("ip") if isinstance(host_data, dict) else None
            host = SEPSesHost(
                id=f"{self.base_uri}Host/{host_name or 'unknown'}",
                hostname=host_name,
                ip=host_ip
            )
        
        # Build Process entity
        process_data = source.get("process", {})
        process = None
        if process_data and isinstance(process_data, dict):
            process = SEPSesProcess(
                id=f"{self.base_uri}Process/{process_data.get('pid', 'unknown')}",
                pid=process_data.get("pid"),
                name=process_data.get("name"),
                command_line=process_data.get("command_line")
            )
        
        # Build User entity
        user_data = source.get("user", {})
        user = None
        if user_data:
            user_name = user_data.get("name") if isinstance(user_data, dict) else str(user_data)
            user = SEPSesUser(
                id=f"{self.base_uri}User/{user_name or 'unknown'}",
                username=user_name
            )
        
        # Build File entity
        file_data = source.get("file", {})
        file_obj = None
        if file_data and isinstance(file_data, dict):
            file_obj = SEPSesFile(
                id=f"{self.base_uri}File/{file_data.get('path', 'unknown')}",
                path=file_data.get("path"),
                name=file_data.get("name")
            )
        
        # Build NetworkFlow entity
        network_flow = None
        src_ip = source.get("source", {}).get("ip") if isinstance(source.get("source"), dict) else None
        dst_ip = source.get("destination", {}).get("ip") if isinstance(source.get("destination"), dict) else None
        if src_ip or dst_ip:
            network_flow = SEPSesNetworkFlow(
                id=f"{self.base_uri}NetworkFlow/{doc_id}",
                src_ip=src_ip,
                src_port=source.get("source", {}).get("port") if isinstance(source.get("source"), dict) else None,
                dst_ip=dst_ip,
                dst_port=source.get("destination", {}).get("port") if isinstance(source.get("destination"), dict) else None
            )
        
        # Extract ancestors (for chain reconstruction)
        ancestors_raw = source.get("kibana.alert.ancestors", [])
        ancestors = []
        if isinstance(ancestors_raw, list):
            for anc in ancestors_raw:
                if isinstance(anc, dict) and "id" in anc:
                    ancestors.append(anc["id"])
        
        return SEPSesLogEvent(
            id=event_uri,
            timestamp=timestamp,
            severity=source.get("kibana.alert.severity"),
            risk_score=source.get("kibana.alert.risk_score"),
            rule_name=source.get("kibana.alert.rule.name"),
            workflow_status=source.get("kibana.alert.workflow_status"),
            elastic_uuid=source.get("kibana.alert.uuid"),
            host=host,
            user=user,
            process=process,
            file=file_obj,
            network_flow=network_flow,
            ancestors=ancestors,
            ttps=ttps
        )
    
    def transform_attack_chain(self, bucket: Dict[str, Any]) -> AttackChain:
        """Transform Elasticsearch aggregation bucket into AttackChain"""
        bucket_key = bucket.get("key", "unknown")
        hits = bucket.get("alerts_in_chain", {}).get("hits", {}).get("hits", [])
        
        # Transform all alerts in chain to SEPSesLogEvent
        log_events = [self.transform_log_event(hit) for hit in hits]
        
        # Extract unique TTPs across the chain
        ttp_set = set()
        unique_ttps = []
        for event in log_events:
            for ttp in event.ttps:
                key = (ttp.technique_id, ttp.tactic_id)
                if key not in ttp_set:
                    ttp_set.add(key)
                    unique_ttps.append(ttp)
        
        # Build chain metadata
        first_ts = min(e.timestamp for e in log_events) if log_events else None
        last_ts = max(e.timestamp for e in log_events) if log_events else None
        
        return AttackChain(
            chain_id=f"chain-{bucket_key}",
            log_events=log_events,
            size=len(log_events),
            first_timestamp=first_ts,
            last_timestamp=last_ts,
            unique_ttps=unique_ttps
        )