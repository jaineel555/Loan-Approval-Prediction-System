$(document).ready(function() {
    $('#submitButton').click(function() {
        // Get form data
        var dependents = $('#dependents').val();
        var education = $('#education').val();
        var employment = $('#employment').val();
        var income = $('#income').val();
        var loanAmount = $('#loanAmount').val();
        var creditScore = $('#creditScore').val();
        var residentialAssets = $('#residentialAssets').val();
        var bankAssets = $('#bankAssets').val();

        // Check if any field is empty
        if (!dependents || !education || !employment || !income || !loanAmount || !creditScore || !residentialAssets || !bankAssets) {
            alert("Please fill all the fields.");
            return;
        }

        // Send data to Flask backend for prediction
        $.ajax({
            type: "POST",
            url: "/predict",  // Flask route for prediction
            data: {
                dependents: dependents,
                education: education,
                employment: employment,
                income: income,
                loanAmount: loanAmount,
                creditScore: creditScore,
                residentialAssets: residentialAssets,
                bankAssets: bankAssets
            },
            success: function(response) {
                // Assuming the response contains the result field
                $('#result').text(response.result);
                alert(response.result);
            },
            error: function(xhr, status, error) {
                console.error(error);
                alert("There was an error processing your request.");
            }
        });
    });
});