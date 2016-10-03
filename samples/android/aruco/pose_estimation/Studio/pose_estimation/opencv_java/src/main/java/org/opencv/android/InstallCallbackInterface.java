package org.opencv.android;

/**
 * Installation callback interface.
 */
public interface InstallCallbackInterface
{
    /**
     * New package installation is required.
     */
    int NEW_INSTALLATION = 0;
    /**
     * Current package installation is in progress.
     */
    int INSTALLATION_PROGRESS = 1;

    /**
     * Target package name.
     * @return Return target package name.
     */
    String getPackageName();
    /**
     * Installation is approved.
     */
    void install();
    /**
     * Installation is canceled.
     */
    void cancel();
    /**
     * Wait for package installation.
     */
    void wait_install();
}
