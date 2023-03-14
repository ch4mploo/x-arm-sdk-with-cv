import tracemalloc, json
import streamlit as st
import gc

class TraceMemory:
    
    def __init__(self) -> None:
        self._TRACES = None

    # @st.cache_resource
    def _init_tracking_object(self):
        tracemalloc.start(10)

        self._TRACES = {
            "runs": 0,
            "tracebacks": {}
        }

    def traceback_exclude_filter(self,patterns, tracebackList):
        """
        Returns False if any provided pattern exists in the filename of the traceback,
        Returns True otherwise.
        """
        for t in tracebackList:
            for p in patterns:
                if p in t.filename:
                    return False
            return True


    def traceback_include_filter(self,patterns, tracebackList):
        """
        Returns True if any provided pattern exists in the filename of the traceback,
        Returns False otherwise.
        """
        for t in tracebackList:
            for p in patterns:
                if p in t.filename:
                    return True
        return False


    def check_for_leaks(self,diff):
        """
        Checks if the same traceback appears consistently after multiple runs.

        diff - The object returned by tracemalloc#snapshot.compare_to
        """
        self._TRACES["runs"] = self._TRACES["runs"] + 1
        tracebacks = set()

        for sd in diff:
            for t in sd.traceback:
                tracebacks.add(t)

        if "tracebacks" not in self._TRACES or len(self._TRACES["tracebacks"]) == 0:
            for t in tracebacks:
                self._TRACES["tracebacks"][t] = 1
        else:
            oldTracebacks = self._TRACES["tracebacks"].keys()
            intersection = tracebacks.intersection(oldTracebacks)
            evictions = set()
            for t in self._TRACES["tracebacks"]:
                if t not in intersection:
                    evictions.add(t)
                else:
                    self._TRACES["tracebacks"][t] = self._TRACES["tracebacks"][t] + 1

            for t in evictions:
                del self._TRACES["tracebacks"][t]

        if self._TRACES["runs"] > 1:
            st.write(f'After {self._TRACES["runs"]} runs the following traces were collected.')
            prettyPrint = {}
            for t in self._TRACES["tracebacks"]:
                prettyPrint[str(t)] = self._TRACES["tracebacks"][t]
            st.write(json.dumps(prettyPrint, sort_keys=True, indent=4))


    def compare_snapshots(self):
        """
        Compares two consecutive snapshots and tracks if the same traceback can be found
        in the diff. If a traceback consistently appears during runs, it's a good indicator
        for a memory leak.
        """
        snapshot = tracemalloc.take_snapshot()
        if "snapshot" in self._TRACES:
            diff = snapshot.compare_to(self._TRACES["snapshot"], "lineno")
            diff = [d for d in diff if
                    d.count_diff > 0 and self.traceback_exclude_filter(["tornado"], d.traceback)
                    and self.traceback_include_filter(["streamlit"], d.traceback)
                    ]
            self.check_for_leaks(diff)

        self._TRACES["snapshot"] = snapshot