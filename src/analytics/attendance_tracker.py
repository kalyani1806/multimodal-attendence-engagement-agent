"""
EPIC 5: Analytics & Recommendation Engine - Sustained Presence Attendance Tracking
Satisfies Scenario 1: Rejects momentary walk-bys and marks attendance based on verified sustained presence.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from pathlib import Path
import json

class AttendanceTracker:
    """
    Manages automated classroom attendance logs based on sustained presence verification.
    """
    def __init__(self, roster_path: Optional[str] = None, min_sustained_seconds: float = 12.0):
        self.min_sustained_seconds = min_sustained_seconds
        self.roster: Dict[str, Any] = {}
        self.students_list: List[Dict[str, str]] = []
        
        # Attendance state per student roll/id:
        # id -> {first_seen, last_seen, active_seconds, is_sustained, status}
        self.records: Dict[str, Dict[str, Any]] = {}
        
        if roster_path and Path(roster_path).exists():
            self.load_roster(roster_path)
            
    def load_roster(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.roster = data
            self.students_list = data.get("students", [])
            for s in self.students_list:
                self.records[s["id"]] = {
                    "id": s["id"],
                    "name": s["name"],
                    "roll_no": s["roll_no"],
                    "first_seen": None,
                    "last_seen": None,
                    "total_active_seconds": 0.0,
                    "sustained_confirmed": False,
                    "status": "ABSENT"
                }

    def update_presence(self, active_participants: List[Dict[str, Any]], session_elapsed_sec: float, dt: float = 1.0):
        """
        Updates presence records from active visual detections in current frame/timestep.
        """
        for p in active_participants:
            pid = p.get("id", 1)
            # Map tracking ID to roster student if available
            student_id = f"STU-{pid:03d}"
            
            if student_id not in self.records:
                self.records[student_id] = {
                    "id": student_id,
                    "name": p.get("name", f"Student {pid}"),
                    "roll_no": f"21CS{pid:02d}",
                    "first_seen": session_elapsed_sec,
                    "last_seen": session_elapsed_sec,
                    "total_active_seconds": 0.0,
                    "sustained_confirmed": False,
                    "status": "WALK-BY (UNVERIFIED)"
                }

            rec = self.records[student_id]
            if rec["first_seen"] is None:
                rec["first_seen"] = session_elapsed_sec
            rec["last_seen"] = session_elapsed_sec
            rec["total_active_seconds"] += dt

            # Verify sustained presence (Scenario 1)
            # Must accumulate at least min_sustained_seconds to rule out walk-by false positives
            if rec["total_active_seconds"] >= self.min_sustained_seconds:
                rec["sustained_confirmed"] = True
                rec["status"] = "PRESENT"
            else:
                rec["status"] = f"ACCUMULATING ({rec['total_active_seconds']:.0f}s/{self.min_sustained_seconds:.0f}s)"

    def finalize_session(self, total_session_sec: float) -> pd.DataFrame:
        """
        Finalizes attendance statuses based on overall session presence ratio.
        """
        rows = []
        for s_id, rec in self.records.items():
            active_sec = rec["total_active_seconds"]
            ratio = active_sec / max(1.0, total_session_sec)
            
            if not rec["sustained_confirmed"] or active_sec < self.min_sustained_seconds:
                final_status = "ABSENT (Walk-By / Below Min Time)"
            elif ratio >= 0.55:
                final_status = "PRESENT"
            elif ratio >= 0.30:
                final_status = "LATE / PARTIAL"
            else:
                final_status = "ABSENT (Insufficient Attendance)"

            rows.append({
                "Student ID": rec["id"],
                "Student Name": rec["name"],
                "Roll No": rec["roll_no"],
                "First Seen (s)": f"{rec['first_seen']:.1f}" if rec['first_seen'] is not None else "--",
                "Last Seen (s)": f"{rec['last_seen']:.1f}" if rec['last_seen'] is not None else "--",
                "Active Duration (s)": f"{rec['total_active_seconds']:.1f}",
                "Presence Ratio": f"{ratio * 100:.1f}%",
                "Attendance Status": final_status
            })
            
        return pd.DataFrame(rows)

    def get_summary_stats(self, total_session_sec: float) -> Dict[str, Any]:
        df = self.finalize_session(total_session_sec)
        total = len(df)
        present = sum(1 for s in df["Attendance Status"] if s == "PRESENT")
        partial = sum(1 for s in df["Attendance Status"] if "PARTIAL" in s)
        absent = total - present - partial
        
        return {
            "total_students": total,
            "present_count": present,
            "partial_count": partial,
            "absent_count": absent,
            "attendance_rate_pct": float(np.round((present / max(1, total)) * 100, 1))
        }
