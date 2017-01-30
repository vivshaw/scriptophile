name := "scriptophile"

organization := "io.github.vivshaw"

version := "1.0-SNAPSHOT"

scalaVersion := "2.10.4"

libraryDependencies  ++= Seq(
  "org.scalanlp" %% "breeze" % "0.12",
  "org.scalanlp" %% "breeze-natives" % "0.12"
)

resolvers ++= Seq(
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)