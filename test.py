import numpy as np
from sigmoids import get_alignment_strength, DEFAULT_PDICT

# NOTE: Testing that higher alignment strengths give smaller misalignment angles is done
# In the relevant places (halotools and modular_alignments)
# Here, I just test that the proper relationships change the alignment strengths in the way I expect

##### REDSHIFT ##########################################################################
def test_redshift_relationship_k():
    # For a given color and log stellar mass, later redshifts (lower number) should have
    # stronger alignments with the defailt dictionary parameters
    params = dict(DEFAULT_PDICT)
    z = np.array([0, 0.5, 1, 1.5, 2])
    color = 0
    logsm = 12
    mu = get_alignment_strength( logsm, color, z, **params )

    # Earlier redshifts are weaker than later (earlier being farther from 0)
    for i in range(len(mu))[1:]:
        assert( mu[i] < mu[i-1] )

    # If I switch the sign of redhsift_k, the opposite should be true
    params["redshift_k"] *= -1
    mu = get_alignment_strength( logsm, color, z, **params )

    # Now, earlier redshifts are stronger than later (earlier being farther from 0)
    for i in range(len(mu))[1:]:
        assert( mu[i] > mu[i-1] )

def test_redshift_relationship_x0():
    # Test that by changing redshift_x0, corresponding values will shift
    # Check mus based on logsm since that will be the final parameter used to get mu
    # Shifting color_xo should affect the final sigmoid
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["redshift_x0"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    # Now shift x0 to the right (higher z)
    params["redshift_x0"] = 1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    # Because the default redshift_k makes it so mu decreases as z increases,
    # this shift in x0 to the right should raise every mu with respect
    # to its original value
    for i in range(len(mu1)):
        assert( mu1[i] < mu2[i] )

def test_redshift_relationship_y_low():
    # Lowering y_low should lower all the values
    # Check that mus calculated at one y_low are higher than calculated
    # at a lower y_low. Check final mu values for a range of logsm
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["redshift_y_low"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    params["redshift_y_low"] = -1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

def test_redshift_relationship_y_high():
    # Rasing y_high should raise all the values
    # Check that mus calculated at one y_high are higher than calculated
    # at a higher y_high. Check final mu values for a range of logsm
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["redshift_y_high"] = 1
    mu1 = get_alignment_strength(logsm, color, z, **params)
    params["redshift_y_high"] = 0.5
    mu2 = get_alignment_strength(logsm, color, z, **params)

    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

##### COLOR #############################################################################
def test_color_relationship_k():
    # For a given redshift and mass, bluer colors should have weaker alignment_strengths
    # using the default parameters
    params = dict(DEFAULT_PDICT)
    z = 0
    color = np.array([-2, -1, 0, 1, 2])
    logsm = 12
    mu = get_alignment_strength( logsm, color, z, **params )

    # Bluer (more negative color) should be weaker
    for i in range(len(mu))[1:]:
        assert( mu[i] > mu[i-1] )

    # Flip the color_k and the reverse should be true
    params["color_k"] *= -1
    mu = get_alignment_strength( logsm, color, z, **params )

    # Now, bluer (more negative color) should be stronger
    for i in range(len(mu))[1:]:
        assert( mu[i] < mu[i-1] )

def test_color_relationship_x0():
    # Test that by changing color_x0, corresponding values will shift
    # Check mus based on logsm since that will be the final parameter used to get mu
    # Shifting color_xo should affect the final sigmoid
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["color_x0"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    # Now shift x0 to the right (higher z)
    params["color_x0"] = 1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    # Since the relationship in general is blue = weak, red = strong,
    # shifting x0 to the right should weaken each mu
    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

def test_color_relationship_y_low():
    # Lowering y_low should lower all the values
    # Check that mus calculated at one y_low are higher than calculated
    # at a lower y_low. Check final mu values for a range of logsm
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["color_y_low"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    params["color_y_low"] = -1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

##### MASS ##############################################################################
def test_mass_relationship_k():
    # For a given redshift and color, lower mass should have weaker alignment_strengths
    # using the default parameters
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array( [ 10, 11, 12, 13, 14 ] )
    mu = get_alignment_strength( logsm, color, z, **params )

    # lower mass should be weaker
    for i in range(len(mu))[1:]:
        assert( mu[i] > mu[i-1] )

    # Flip the logsm_k and the reverse should be true
    params["logsm_k"] *= -1
    mu = get_alignment_strength( logsm, color, z, **params )

    # Now, lower mass should be stronger
    for i in range(len(mu))[1:]:
        assert( mu[i] < mu[i-1] )

def test_mass_relationship_x0():
    # Test that by changing redshift_x0, corresponding values will shift
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["logsm_x0"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    # Now shift x0 to the right (higher z)
    params["logsm_x0"] = 1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    # Since the relationship in general is low mass = weak, high mass = strong,
    # shifting x0 to the right should weaken each mu
    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

def test_mass_relationship_y_low():
    # Lowering y_low should lower all the values
    # Check that mus calculated at one y_low are higher than calculated
    # at a lower y_low. Check final mu values for a range of logsm
    params = dict(DEFAULT_PDICT)
    z = 0
    color = 0
    logsm = np.array([10,11,12,13,14])

    params["logsm_y_low"] = 0
    mu1 = get_alignment_strength(logsm, color, z, **params)
    params["logsm_y_low"] = -1
    mu2 = get_alignment_strength(logsm, color, z, **params)

    for i in range(len(mu1)):
        assert( mu1[i] > mu2[i] )

##### MAIN ##############################################################################

if __name__ == "__main__":
    print("Testing")
    
    test_redshift_relationship_k()
    test_redshift_relationship_x0()
    test_redshift_relationship_y_low()
    test_redshift_relationship_y_high()

    test_color_relationship_k()
    test_color_relationship_x0()
    test_color_relationship_y_low()

    test_mass_relationship_k()
    test_mass_relationship_x0()
    test_mass_relationship_y_low()

    print("Done Testing")