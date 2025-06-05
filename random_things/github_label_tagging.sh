#!/opt/homebrew/bin/bash

# GitHub Label Standardizer Script
# Usage: ./github_label_tagging.sh napistu repo1 repo2 repo3 ...

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ORG_NAME> <repo1> [repo2] [repo3] ..."
    echo "Example: $0 myorg frontend-app backend-api docs"
    exit 1
fi

ORG_NAME="$1"
shift  # Remove first argument, leaving only repo names
REPOS=("$@")

# Define standard labels with colors and descriptions
declare -A STANDARD_LABELS=(
    ["bug"]="d73a4a:Something isn't working"
    ["minor_bug"]="f85149:Minor issue that doesn't break functionality"
    ["tech_debt"]="8b6914:Lower priority, nice-to-have improvements"
    ["good_first_issue"]="7057ff:Good for newcomers"
    ["invalid"]="e4e669:This doesn't seem right"
    ["wontfix"]="ffffff:This will not be worked on"
    ["question"]="d876e3:Further information is requested"
    ["enhancement_biology"]="0075ca:New features affecting data sources and representations of biology"
    ["enhancement_operations"]="1f4e79:Features affecting operation and validation of codebase"
    ["discussion"]="fbca04:Open discussion, brainstorming, design decisions"
    ["analysis"]="006b75:Statistical analysis, algorithms, computational methods"
)

# Map internal names to display names
declare -A LABEL_DISPLAY_NAMES=(
    ["bug"]="bug"
    ["minor_bug"]="bug: minor"
    ["tech_debt"]="tech debt"
    ["good_first_issue"]="good first issue"
    ["invalid"]="invalid"
    ["wontfix"]="wontfix"
    ["question"]="question"
    ["enhancement_biology"]="enhancement: biology"
    ["enhancement_operations"]="enhancement: operations"
    ["discussion"]="discussion"
    ["analysis"]="analysis"
)

echo "🏷️  GitHub Label Standardizer"
echo "Organization: $ORG_NAME"
echo "Repositories: ${REPOS[*]}"
echo "Standard labels: ${!STANDARD_LABELS[*]}"
echo ""

# Function to process a single repository
process_repo() {
    local repo="$1"
    local full_repo="$ORG_NAME/$repo"
    
    echo "📁 Processing repository: $repo"
    
    # Check if repo exists and is accessible
    if ! gh repo view "$full_repo" >/dev/null 2>&1; then
        echo "❌ Error: Cannot access repository $full_repo"
        return 1
    fi
    
    # Get current labels
    echo "  📋 Getting current labels..."
    current_labels=$(gh label list --repo "$full_repo" --json name -q '.[].name' 2>/dev/null)
    
    if [ $? -ne 0 ]; then
        echo "  ❌ Error: Failed to get labels for $full_repo"
        return 1
    fi
    
    # Remove labels that are not in our standard set
    echo "  🧹 Removing non-standard labels..."
    echo "$current_labels" | while IFS= read -r label; do
        if [[ -n "$label" ]]; then
            # Check if this label matches any of our display names
            label_found=false
            for key in "${!LABEL_DISPLAY_NAMES[@]}"; do
                if [[ "${LABEL_DISPLAY_NAMES[$key]}" == "$label" ]]; then
                    label_found=true
                    break
                fi
            done
            
            if [[ "$label_found" == false ]]; then
                echo "    🗑️  Deleting: $label"
                gh label delete "$label" --repo "$full_repo" --yes 2>/dev/null || echo "    ⚠️  Could not delete: $label"
            fi
        fi
    done
    
    # Add/update standard labels
    echo "  ➕ Adding/updating standard labels..."
    for label_key in "${!STANDARD_LABELS[@]}"; do
        label_name="${LABEL_DISPLAY_NAMES[$label_key]}"
        IFS=':' read -r color description <<< "${STANDARD_LABELS[$label_key]}"
        echo "    🏷️  $label_name"
        
        # Try to create the label (will fail if it exists)
        if gh label create "$label_name" --color "$color" --description "$description" --repo "$full_repo" 2>/dev/null; then
            echo "    ✅ Created: $label_name"
        else
            # If creation failed, try to edit existing label
            if gh label edit "$label_name" --color "$color" --description "$description" --repo "$full_repo" 2>/dev/null; then
                echo "    ✏️  Updated: $label_name"
            else
                echo "    ❌ Failed to create/update: $label_name"
            fi
        fi
    done
    
    echo "  ✅ Completed: $repo"
    echo ""
}

# Process each repository
for repo in "${REPOS[@]}"; do
    process_repo "$repo"
done

echo "🎉 Label standardization complete!"
echo ""
echo "📊 Summary:"
echo "  • Processed ${#REPOS[@]} repositories"
echo "  • Applied ${#STANDARD_LABELS[@]} standard labels"
echo "  • Removed all non-standard labels"