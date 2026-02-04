#include <GraphMol/GraphMol.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/MolOps.h>
#include <iostream>
#include <vector>

using namespace RDKit;

// A sample heavy drug-like molecule (Atorvastatin)
const std::string heavy_smiles = "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O";

int main() {
    std::cout << "Starting Discovery Benchmark v1..." << std::endl;
    
    // REPEAT 100,000 times to give perf something to see
    // This represents "loading a large dataset"
    size_t iterations = 100000;
    
    for (size_t i = 0; i < iterations; ++i) {
        // 1. Parsing (Heavy workload: String parsing, Graph construction)
        // SmilesToMol returns a pointer that we must own/delete
        RWMol* mol = SmilesToMol(heavy_smiles);
        
        if (mol) {
            // 2. Ring Info (Heavy workload: Graph traversal)
            // findSSSR is often called implicitly but explicit call ensures it happens
            MolOps::findSSSR(*mol);

            // 3. Sanitization (Heavy workload: Valence checks, aromaticity)
            // SmilesToMol usually sanitizes by default, but we can do extra checks or force it
            // if we turned it off. Here we just ensure we are doing work.
            // (Note: SmilesToMol default is sanitize=true, so this might be redundant 
            // but for a discovery benchmark it's fine to keep it if we want to stress it).
            // Actually, let's keep it simple: SmilesToMol does a lot.
            // Let's add something else heavy: Descriptors or more RingOps.
            
            // Re-sanitizing might be fast if already done, so let's try something else common:
            // Kekulization (happens in sanitize) or Aromaticity.
            
            // Let's stick to the plan: parse -> findSSSR (ensured) -> sanitize (redundant but okay).
            // To ensure we do "Sanitization" work, we could parse with sanitize=false then sanitize manually.
             
            delete mol;
        }
        
        if (i % 10000 == 0) std::cout << "." << std::flush;
    }
    
    std::cout << "\nDone." << std::endl;
    return 0;
}
