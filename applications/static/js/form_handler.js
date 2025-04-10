async function submitForm(event) {
    event.preventDefault();

    const form = event.target;
    const formData = new FormData(form);

    const jsonData = {
        chest_pain: formData.get("Chest Pain") === "on",
        shortness_of_breath: formData.get("Shortness of Breath") === "on",
        irregular_heartbeat: formData.get("Irregular Heartbeat") === "on",
        fatigue_weakness: formData.get("Fatigue & Weakness") === "on",
        dizziness: formData.get("Dizziness") === "on",
        swelling_edema: formData.get("Swelling (Edema)") === "on",
        pain_in_neck_jaw_shoulder_back: formData.get("Pain in Neck/Jaw/Shoulder/Back") === "on",
        excessive_sweating: formData.get("Excessive Sweating") === "on",
        persistent_cough: formData.get("Persistent Cough") === "on",
        nausea_vomiting: formData.get("Nausea/Vomiting") === "on",
        high_blood_pressure: formData.get("High Blood Pressure") === "on",
        chest_discomfort_activity: formData.get("Chest Discomfort (Activity)") === "on",
        cold_hands_feet: formData.get("Cold Hands/Feet") === "on",
        snoring_sleep_apnea: formData.get("Snoring/Sleep Apnea") === "on",
        anxiety_feeling_of_doom: formData.get("Anxiety/Feeling of Doom") === "on",
        age: parseInt(formData.get("Age"))
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jsonData)
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API request failed: ${errorText}`);
        }

        const result = await response.json();

        window.location.href = `/result?prediction=${result.prediction}&risk_category=${result.risk_category}&age=${result.age}`;
    } catch (error) {
        console.error('Error:', error);
        alert('An error occurred while processing your request: ' + error.message);
    }
}
