function preprocessTextAdvanced(text) {
    const numberMatches = text.match(/\b\d+(?:\.\d+)*(?:%|percent|million|billion|thousand)?\b/g) || [];
    let normalized = text.toLowerCase()
        .replace(/['']/g, "'")
        .replace(/[^a-z0-9\s]/g, '')

}
