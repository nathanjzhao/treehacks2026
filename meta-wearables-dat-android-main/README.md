# Meta Wearables Device Access Toolkit for Android

[![Maven](https://img.shields.io/badge/Maven-0.4.0-brightgreen?logo=apachemaven)](https://github.com/orgs/facebook/packages?repo_name=meta-wearables-dat-android)
[![Docs](https://img.shields.io/badge/API_Reference-0.4-blue?logo=meta)](https://wearables.developer.meta.com/docs/reference/android/dat/0.4)

The Meta Wearables Device Access Toolkit enables developers to utilize Meta's AI glasses to build hands-free wearable experiences into their mobile applications.
By integrating this SDK, developers can reliably connect to Meta's AI glasses and leverage capabilities like video streaming and photo capture.

The Wearables Device Access Toolkit is in developer preview.
Developers can access our SDK and documentation, test on supported AI glasses, and create organizations and release channels to share with test users.

## Documentation & Community

Find our full [developer documentation](https://wearables.developer.meta.com/docs/develop/) on the Wearables Developer Center.

You can find an overview of the Wearables Developer Center [here](https://wearables.developer.meta.com/).
Create an account to stay informed of all updates, report bugs and register your organization.
Set up a project and release channel to share your integration with test users.

For help, discussion about best practices or to suggest feature ideas visit our [discussions forum](https://github.com/facebook/meta-wearables-dat-android/discussions).

See the [changelog](CHANGELOG.md) for the latest updates.

## Including the SDK in your project

You can add the Wearables Device Access Toolkit to your Gradle project by following the steps below.
You will need to provide a personal access token (classic) with at least **read:packages** scope as an environment variable named `GITHUB_TOKEN` or
by adding it as a property named `github_token` in your `local.properties` file.
See [SDK for Android setup](https://wearables.developer.meta.com/docs/getting-started-toolkit/#sdk-for-android-setup) for more details.

### 1. Add the repository definition to `settings.gradle.kts`

```kotlin
val localProperties =
    Properties().apply {
        val localPropertiesPath = rootDir.toPath() / "local.properties"
        if (localPropertiesPath.exists()) {
            load(localPropertiesPath.inputStream())
        }
    }

dependencyResolutionManagement {
    ...
    repositories {
        ...
        maven {
            url = uri("https://maven.pkg.github.com/facebook/meta-wearables-dat-android")
            credentials {
                username = "" // not needed
                password = System.getenv("GITHUB_TOKEN") ?: localProperties.getProperty("github_token")
            }
        }
    }
}
```

### 2. Declare the Wearables Device Access Toolkit artifacts in `libs.versions.toml`

Check the available versions in [GitHub Packages](https://github.com/orgs/facebook/packages?repo_name=meta-wearables-dat-android).

```toml
[versions]
mwdat = "0.4.0"

[libraries]
mwdat-core = { group = "com.meta.wearable", name = "mwdat-core", version.ref = "mwdat" }
mwdat-camera = { group = "com.meta.wearable", name = "mwdat-camera", version.ref = "mwdat" }
mwdat-mockdevice = { group = "com.meta.wearable", name = "mwdat-mockdevice", version.ref = "mwdat" }
```

### 3. Add the required components as dependencies in your app's `build.gradle.kts`

```kotlin
dependencies {
    implementation(libs.mwdat.core)
    implementation(libs.mwdat.camera)
    implementation(libs.mwdat.mockdevice)
}
```

## Developer Terms

- By using the Wearables Device Access Toolkit, you agree to our [Meta Wearables Developer Terms](https://wearables.developer.meta.com/terms),
  including our [Acceptable Use Policy](https://wearables.developer.meta.com/acceptable-use-policy).
- By enabling Meta integrations, including through this SDK, Meta may collect information about how users' Meta devices communicate with your app.
  Meta will use this information collected in accordance with our [Privacy Policy](https://www.meta.com/legal/privacy-policy/).
- You may limit Meta's access to data from users' devices by following the instructions below.

### Opting out of data collection

To configure analytics settings in your Meta Wearables DAT Android app, add the following `<meta-data>` element to your
app's `AndroidManifest.xml` file within the `<application>` element:

```xml
<meta-data
    android:name="com.meta.wearable.mwdat.ANALYTICS_OPT_OUT"
    android:value="true"
    />
```

**Default behavior:** If the `ANALYTICS_OPT_OUT` metadata is missing or set to `false`, analytics are enabled
(i.e., you are **not** opting out). Set to `true` to disable data collection.

**Note:** In other words, this setting controls whether or not you're opting out of analytics:

- `true` = Opt out (analytics **disabled**)
- `false` = Opt in (analytics **enabled**)

**Complete example:**

```xml
<application
    android:name=".MyApplication"
    android:label="MyApp"
    android:icon="@mipmap/app_launcher">

    <!-- Required: Your application ID from Wearables Developer Center -->
    <meta-data
        android:name="com.meta.wearable.mwdat.APPLICATION_ID"
        android:value="your_app_id_here"
        />

    <!-- Optional: Disable analytics -->
    <meta-data
        android:name="com.meta.wearable.mwdat.ANALYTICS_OPT_OUT"
        android:value="true"
        />

    <!-- Your activities and other components -->
</application>
```

## License

See the [LICENSE](LICENSE) file.
