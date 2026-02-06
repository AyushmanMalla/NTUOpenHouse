import xir
import vart

#Replace the path below as required -> it will throw a core dumped error if it cannot find the model
MODEL_PATH = "/home/ubuntu/Kria-PYNQ/movenet_kr260_vai25.xmodel"

def get_dpu_subgraph(graph):
    """
    Recursively find the subgraph designated for the DPU.
    """
    root = graph.get_root_subgraph()
    
    # Helper function to traverse children
    def find_dpu(subgraph):
        if subgraph.has_attr("device"):
            if subgraph.get_attr("device").upper() == "DPU":
                return subgraph
        
        for child in subgraph.get_children():
            result = find_dpu(child)
            if result is not None:
                return result
        return None

    return find_dpu(root)

def inspect_model(path):
    print(f"Loading model: {path}")
    # Deserialize the graph
    g = xir.Graph.deserialize(path)
    
    # Find the DPU subgraph using the robust helper
    dpu_subgraph = get_dpu_subgraph(g)
    
    if dpu_subgraph is None:
        print("Error: Could not find DPU subgraph in xmodel.")
        print("Available subgraphs:")
        for child in g.get_root_subgraph().get_children():
            print(f" - {child.get_name()} (Device: {child.get_attr('device') if child.has_attr('device') else 'CPU/None'})")
        return None

    print(f"Found DPU Subgraph: {dpu_subgraph.get_name()}")

    # Create a runner to inspect tensors
    # Note: 'runner' variable must be kept alive or VART might clean it up
    runner = vart.Runner.create_runner(dpu_subgraph, "run")
    
    # INSPECT INPUTS
    input_tensors = runner.get_input_tensors()
    print("\n--- Input Tensors ---")
    for t in input_tensors:
        print(f"Name: {t.name}")
        print(f"Shape: {t.dims}") # format is typically [Batch, Height, Width, Channels]
        print(f"Type: {t.dtype}")
        print(f"Fix Point Position: {t.get_attr('fix_point')}")
        print("-" * 20)

    # INSPECT OUTPUTS
    output_tensors = runner.get_output_tensors()
    print("\n--- Output Tensors ---")
    for t in output_tensors:
        print(f"Name: {t.name}")
        print(f"Shape: {t.dims}")
        print(f"Fix Point Position: {t.get_attr('fix_point')}")
        print("-" * 20)
        
    return runner, input_tensors, output_tensors

if __name__ == "__main__":
    try:
        runner, inputs, outputs = inspect_model(MODEL_PATH)
        print("\nSUCCESS: Model loaded and inspected.")
    except Exception as e:
        print(f"An error occurred: {e}")